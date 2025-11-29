import sys
import threading
from pathlib import Path
sys.path.insert(0, Path(__file__).parents[2].as_posix())

from tqdm import tqdm
import cv2
import click
import nvtx
from deploy2serve.utils.logger import get_logger
import glob
import torch
import time
from typing import Tuple, List, Optional
from ultralytics.utils.ops import scale_boxes
from ultralytics.utils.ops import non_max_suppression
from torchvision.io import decode_jpeg, ImageReadMode, read_file
from torchvision.utils import draw_bounding_boxes

from core.pool_manager import PoolManager
from utils.env import get_project_root
from concurrent.futures import ThreadPoolExecutor


class YoloProcessor(object):
    def __init__(self, input_shape: Tuple[int, int]) -> None:
        self.input_shape: Tuple[int, int] = input_shape
        self.input: Optional[torch.Tensor] = None

    @staticmethod
    @torch.jit.script
    def letterbox(
        image: torch.Tensor,
        new_shape: Tuple[int, int],
        color: Tuple[int, int, int] = (114, 114, 114),
        auto: bool = False,
        scaleup: bool = True,
        stride: int = 32
    ) -> Tuple[torch.Tensor, float, Tuple[float, float]]:
        # Resize and pad image while meeting stride-multiple constraints
        shape = image.shape[-2:]  # current shape [height, width]

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw = float(new_shape[1] - new_unpad[0])
        dh = float(new_shape[0] - new_unpad[1])

        if auto:
            dw = torch.remainder(torch.tensor(dw), stride).item()
            dh = torch.remainder(torch.tensor(dh), stride).item()

        dw /= 2
        dh /= 2

        top = int(torch.round(torch.tensor(dh - 0.1)).item())
        bottom = int(torch.round(torch.tensor(dh + 0.1)).item())
        left = int(torch.round(torch.tensor(dw - 0.1)).item())
        right = int(torch.round(torch.tensor(dw + 0.1)).item())
        image = image.unsqueeze(0).float()

        if shape[::-1] != new_unpad:
            image = torch.nn.functional.interpolate(
                image,
                size=new_unpad[::-1],
                mode="bilinear",
                align_corners=False
            )

        if left > 0 or right > 0 or top > 0 or bottom > 0:
            image = torch.nn.functional.pad(
                image,
                pad=(left, right, top, bottom),
                mode="constant",
                value=float(color[0]) / 255.0
            )

        return image / 255., r, (dw, dh)

    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        return self.letterbox(image, self.input_shape)[0]

    def postprocess(
        self, output: torch.Tensor, orig_shape
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        boxes: List[torch.Tensor] = []
        scores: List[torch.Tensor] = []
        classes: List[torch.Tensor] = []
        for idx in range(len(output)):
            detections = non_max_suppression(output[idx])[0]
            boxes.append(scale_boxes(self.input_shape, detections[:, :4], orig_shape))
            scores.append(detections[:, 4:5].reshape(-1, 1))
            classes.append(detections[:, 5:].reshape(-1).to(torch.int))
        return boxes, scores, classes


def threaded_read(frames: List[str], num_workers: int = 8, batch_size: int = 50) -> List[torch.Tensor]:
    total_frames = len(frames)
    images: List[torch.Tensor] = []

    with tqdm(total=total_frames, desc="Loading frames") as pbar:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for i in range(0, total_frames, batch_size):
                batch_paths = frames[i: i + batch_size]
                batch_images = list(executor.map(lambda path: read_file(path), batch_paths))
                images.extend(batch_images)
                pbar.update(len(batch_images))
    return images


@click.command()
@click.option("--frames-folder", default=f"{get_project_root()}/images", type=str,
              help="Path to file which consider labels in coco format.")
@click.option("--no-preview", is_flag=True, default=True,
              help="Disable to show results.")
def simple_launch(frames_folder, no_preview):
    logger = get_logger("benchmark")
    processor = YoloProcessor((384, 640))

    with nvtx.annotate("create pool object"):
        pool = PoolManager(
            model_path=f"{get_project_root()}/checkpoints/yolo/tensorrt/model.plan",
            input_shapes={"images": (1, 3, 384, 640), "output": (1, 84, 5040)},
            num_workers=1,
            device="cuda:0",
            streams_per_worker=1,
            asynchronous=True,
            use_graph=True,
            use_unique_context=True,
            mutex=threading.Lock()
        )

    try:
        frames = glob.glob(f"{frames_folder}/*")[:100]

        with nvtx.annotate("read files"):
            batch_size = 50
            images = threaded_read(frames, num_workers=max(1, len(frames) // batch_size), batch_size=batch_size)

        with nvtx.annotate("convert raw buffers to cuda tensors"):
            tensors = decode_jpeg(images, mode=ImageReadMode.RGB, device="cuda:0")

        with nvtx.annotate("preprocess for inputs"):
            inputs = [{'images': processor.preprocess(tensors[idx])} for idx in range(len(tensors))]

        with nvtx.annotate("processing tasks"):
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
                boxes, scores, classes = processor.postprocess(result, tensors[idx].shape[-2:])
                if len(boxes):
                    labels = [f"{classes[0][j].item()}: {round(scores[0][j].item(), ndigits=2)}"
                              for j in range(len(classes[0]))]
                    image = draw_bounding_boxes(
                        tensors[idx], boxes[0], labels, colors=[(0, 233, 255)] * boxes[0].shape[0], width=2
                    )
                    cv2.imshow("", image.permute(1, 2, 0).cpu().numpy())
                    cv2.waitKey(0)

    except KeyboardInterrupt:
        logger.warning("Stopping...")
    except Exception as error:
        logger.critical(error)
    finally:
        pool.stop_all()


if __name__ == "__main__":
    simple_launch()
