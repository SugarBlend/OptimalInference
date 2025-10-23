import sys
from pathlib import Path
sys.path.insert(0, Path(__file__).parents[2].as_posix())

import cv2
import click
import nvtx
from deploy2serve.utils.logger import get_logger
import glob
from mmcv.visualization.image import imshow_det_bboxes
import numpy as np
import torch
import time
from typing import Tuple, List
from ultralytics.data.augment import LetterBox
from ultralytics.utils.ops import scale_boxes
from ultralytics.utils.nms import non_max_suppression

from core.pool_manager import PoolManager


class YoloProcessor(object):
    def __init__(self, input_shape: Tuple[int, int]) -> None:
        self.input_shape: Tuple[int, int] = input_shape
        self.letterbox = LetterBox(new_shape=input_shape)

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        preprocessed = torch.from_numpy(self.letterbox(image=image)).to("cuda:0")
        preprocessed = preprocessed.permute(2, 0, 1)
        preprocessed = preprocessed / 255.0
        preprocessed = preprocessed.half()
        return preprocessed[None]

    def postprocess(
        self, output: torch.Tensor, orig_shape
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        boxes: List[np.ndarray] = []
        scores: List[np.ndarray] = []
        classes: List[np.ndarray] = []
        for idx in range(len(output)):
            detections = non_max_suppression(output[idx])[0]
            boxes.append(scale_boxes(self.input_shape, detections[:, :4], orig_shape).cpu().numpy())
            scores.append(detections[:, 4:5].reshape(-1, 1).cpu().numpy())
            classes.append(detections[:, 5:].reshape(-1).to(torch.int).cpu().numpy())
        return boxes, scores, classes


@click.command()
@click.option("--frames-folder", default="../images", type=str,
              help="Path to file which consider labels in coco format.")
def simple_launch(frames_folder):
    logger = get_logger("benchmark")
    processor = YoloProcessor((384, 640))
    pool = PoolManager(
        model_path="../checkpoints/yolo/tensorrt/model.plan",
        input_shapes={"images": (1, 3, 384, 640), "output": (1, 84, 5040)},
        num_workers=4,
        device="cuda:0",
        streams_per_worker=1,
        asynchronous=True
    )

    try:
        with nvtx.annotate("create placeholders"):
            frames = glob.glob(f"{frames_folder}/*")
            inputs = [{"images": processor.preprocess(cv2.imread(frame))} for frame in frames]

        start_time = time.time()
        worker_task_pairs = pool.submit_batch(inputs)
        results = pool.get_synchronized_results(worker_task_pairs)
        total_time = time.time() - start_time

        successful = len([r for r in results if r is not None])

        logger.info(f"Successful: {successful}/{len(frames)}")
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info(f"Throughput: {successful / total_time:.2f} inferences/sec")
        status = pool.get_pool_status()
        logger.info(f"Pool status: {status}")
        for idx, result in enumerate(results):
            image = cv2.imread(frames[idx])

            boxes, scores, classes = processor.postprocess(result, image.shape[:2])
            if len(boxes):
                imshow_det_bboxes(
                    image.copy(), np.concatenate([boxes[0], scores[0]], axis=1), classes[0],
                    bbox_color=(0, 233, 255), text_color=(0, 233, 255), thickness=2, show=True,
                )

    except KeyboardInterrupt:
        logger.warning("Stopping...")
    except Exception as error:
        logger.critical(error)
    finally:
        pool.stop_all()


if __name__ == "__main__":
    simple_launch()
