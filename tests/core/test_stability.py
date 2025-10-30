import sys
from pathlib import Path
sys.path.insert(0, Path(__file__).parents[1].as_posix())

import cv2
import glob
import torch
from utils.benchmark import YoloProcessor, PoolManager
import pytest
import _pytest
from utils.env import get_project_root


class TestExecutionAbility:
    @pytest.mark.parametrize("asynchronous", [True, False])
    @pytest.mark.parametrize("threads_number", [1, 4])
    @pytest.mark.parametrize("streams_per_thread", [1, 4])
    @pytest.mark.repeat(3)
    def test_multi_thread(
        self,
        request: _pytest.fixtures.TopRequest,
        asynchronous: bool,
        threads_number: int,
        streams_per_thread: int
    ):
        current_step, repeats = request.node.callspec.id.split("-")[-2:]

        processor = YoloProcessor((384, 640))
        with PoolManager(
            model_path=f"{get_project_root()}/checkpoints/yolo/tensorrt/model.plan",
            input_shapes={"images": (1, 3, 384, 640), "output": (1, 84, 5040)},
            num_workers=threads_number,
            device="cuda:0",
            streams_per_worker=streams_per_thread,
            asynchronous=asynchronous
        ) as pool:
            frames_folder = "../../images"
            frames = glob.glob(f"{frames_folder}/*")[:1000]
            inputs = [{"images": processor.preprocess(cv2.imread(frame))} for frame in frames]

            worker_task_pairs = pool.submit_batch(inputs)
            results = pool.get_synchronized_results(worker_task_pairs)

            successful = len([r for r in results if r is not None])
            assert successful == len(frames), "Number of successful tries must be the same as number of testing frames."

        request.config._test_results.append(results)
        if len(request.config._test_results) > 1:
            reference_predictions, compare_predictions = request.config._test_results
            for i in range(len(reference_predictions)):
                difference = torch.any(reference_predictions[i][0] - compare_predictions[i][0])
                assert not difference, (
                    f"Detected different frames on same iteration: {int(current_step) - 1}/{repeats} and "
                    f"{current_step}/{repeats}. Frame: {i}. {reference_predictions[i][0]}, {compare_predictions[i][0]}"
                )
        if current_step == repeats:
            request.config._test_results.clear()
        torch.cuda.empty_cache()

    @pytest.mark.parametrize("threads_number", [1, 4])
    @pytest.mark.parametrize("streams_per_thread", [1, 4])
    @pytest.mark.repeat(6)
    def test_modes_similarity(
        self,
        request: _pytest.fixtures.TopRequest,
        threads_number: int,
        streams_per_thread: int
    ):
        current_step, repeats = request.node.callspec.id.split("-")[-2:]

        processor = YoloProcessor((384, 640))
        with PoolManager(
            model_path=f"{get_project_root()}/checkpoints/yolo/tensorrt/model.plan",
            input_shapes={"images": (1, 3, 384, 640), "output": (1, 84, 5040)},
            num_workers=streams_per_thread,
            device="cuda:0",
            streams_per_worker=streams_per_thread,
            asynchronous=int(current_step) % 2 == 0
        ) as pool:
            frames_folder = "../../images"
            frames = glob.glob(f"{frames_folder}/*")[:1000]
            inputs = [{"images": processor.preprocess(cv2.imread(frame))} for frame in frames]

            worker_task_pairs = pool.submit_batch(inputs)
            results = pool.get_synchronized_results(worker_task_pairs)

            successful = len([r for r in results if r is not None])
            assert successful == len(frames), "Number of successful tries must be the same as number of testing frames."

        request.config._test_results.append(results)
        if len(request.config._test_results) > 1:
            reference_predictions, compare_predictions = request.config._test_results
            for i in range(len(reference_predictions)):
                difference = torch.any(reference_predictions[i][0] - compare_predictions[i][0])
                assert not difference, (
                    f"Detected different frames on same iteration: {int(current_step) - 1}/{repeats} and "
                    f"{current_step}/{repeats}. Frame: {i}. {reference_predictions[i][0]}, {compare_predictions[i][0]}"
                )
        if current_step == repeats:
            request.config._test_results.clear()
        torch.cuda.empty_cache()
