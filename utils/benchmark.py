import sys
from pathlib import Path
sys.path.insert(0, Path(__file__).parents[2].as_posix())

import cv2
import click
import nvtx
from deploy2serve.utils.logger import get_logger
import glob
import numpy as np
import torch
import time
from typing import Tuple, List, Optional
from ultralytics.data.augment import LetterBox
from ultralytics.utils.ops import scale_boxes
from ultralytics.utils.ops import non_max_suppression

from core.pool_manager import PoolManager
from utils.env import get_project_root


def imshow_det_bboxes(img: np.ndarray,
                      bboxes: np.ndarray,
                      labels: np.ndarray,
                      class_names: List[str] = None,
                      score_thr: float = 0,
                      bbox_color: Tuple[int, ...] = (0, 255, 0),
                      text_color: Tuple[int, ...] = (0, 255, 0),
                      thickness: int = 1,
                      font_scale: float = 0.5,
                      show: bool = True,
                      win_name: str = '',
                      wait_time: int = 0,
                      out_file: Optional[str] = None):
    """Draw bboxes and class labels (with scores) on an image.

    Args:
        img (str or ndarray): The image to be displayed.
        bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5).
        labels (ndarray): Labels of bboxes.
        class_names (list[str]): Names of each classes.
        score_thr (float): Minimum score of bboxes to be shown.
        bbox_color (Color or str or tuple or int or ndarray): Color
            of bbox lines.
        text_color (Color or str or tuple or int or ndarray): Color
            of texts.
        thickness (int): Thickness of lines.
        font_scale (float): Font scales of texts.
        show (bool): Whether to show the image.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
        out_file (str or None): The filename to write the image.

    Returns:
        ndarray: The image with bboxes drawn on it.
    """
    assert bboxes.ndim == 2
    assert labels.ndim == 1
    assert bboxes.shape[0] == labels.shape[0]
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5
    img = np.ascontiguousarray(img)

    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]

    for bbox, label in zip(bboxes, labels):
        bbox_int = bbox.astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])
        cv2.rectangle(
            img, left_top, right_bottom, bbox_color, thickness=thickness)
        label_text = class_names[
            label] if class_names is not None else f'cls {label}'
        if len(bbox) > 4:
            label_text += f'|{bbox[-1]:.02f}'
        cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 2),
                    cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)

    if show:
        cv2.imshow(win_name, img)
        cv2.waitKey(wait_time)
    if out_file is not None:
        cv2.imwrite(out_file, img)
    return img


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
@click.option("--frames-folder", default=f"{get_project_root()}/images", type=str,
              help="Path to file which consider labels in coco format.")
@click.option("--no-preview", is_flag=True, default=False,
              help="Disable to show results.")
def simple_launch(frames_folder, no_preview):
    logger = get_logger("benchmark")
    processor = YoloProcessor((384, 640))
    pool = PoolManager(
        model_path=f"{get_project_root()}/checkpoints/yolo/tensorrt/model.plan",
        input_shapes={"images": (1, 3, 384, 640), "output": (1, 84, 5040)},
        num_workers=1,
        device="cuda:0",
        streams_per_worker=1,
        asynchronous=True,
        use_graph=True,
    )

    try:
        with nvtx.annotate("create placeholders"):
            frames = glob.glob(f"{frames_folder}/*")[:1000]
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
        if not no_preview:
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
