from collections import deque
from deploy2serve.deployment.models.common import LoggingMeta
from omegaconf import OmegaConf, DictConfig
import time
import torch
import threading
from typing import Dict, List

from core.thread_worker import WorkerThread
from core.override import TensorRTExecutor
from utils.wrappers import nvtx


class PoolManager(object, metaclass=LoggingMeta):
    def __init__(
        self,
        model_path: str,
        input_shapes: Dict[str, tuple],
        num_workers: int = 2,
        device: str = "cuda:0",
        log_level: str = "ERROR",
        enable_nvtx: bool = True,
        streams_per_worker: int = 1,
        mixed_stream_config: List[int] = None,
        asynchronous: bool = False,
        use_graph: bool = False
    ):
        self.model_path: str = model_path
        self.input_shapes: Dict[str, tuple] = input_shapes
        self.num_workers: int = num_workers
        self.device: str = device
        self.log_level: str = log_level
        self.enable_nvtx: bool = enable_nvtx
        self.streams_per_worker: int = streams_per_worker
        self.mixed_stream_config: List[int] = mixed_stream_config
        self.asynchronous: bool = asynchronous
        self.use_graph: bool = use_graph

        self.workers: List[WorkerThread] = []
        self.task_counter = 0
        self.worker_rotation = deque()
        self.lock = threading.RLock()

        self.start_time = time.time()
        self.total_submitted = 0
        self.total_completed = 0

        self._initialize_workers()

    @nvtx.annotate("get_synchronized_results")
    def get_synchronized_results(self, worker_task_pairs: List[tuple], timeout: float = 10.0) -> List:
        completion_events = []
        for worker, task_id in worker_task_pairs:
            event = worker.get_completion_event(task_id)
            if event:
                completion_events.append(event)

        with nvtx.annotate("wait_threads_events", color="red"):
            if completion_events:
                for event in completion_events:
                    torch.cuda.current_stream().wait_event(event)

        with nvtx.annotate("push_results", color="green"):
            results = []
            for worker, task_id in worker_task_pairs:
                with nvtx.annotate(f"waiting_worker_{worker.worker_id}", color="green"):
                    result = worker.get_result()

                if result and result[0] == task_id:
                    results.append(result[1])
                    with self.lock:
                        self.total_completed += 1
                else:
                    results.append(None)

        for worker, task_id in worker_task_pairs:
            worker.cleanup_completion_event(task_id)
        return results

    @nvtx.annotate("initialize_worker_pool")
    def _initialize_workers(self) -> None:
        self.logger.info(
            f"Initializing {self.num_workers} TensorRT workers with {self.streams_per_worker} streams each...")

        self.deserialized_model = TensorRTExecutor.load(
            self.model_path, self.device, self.log_level
        )

        stream_configs = self._get_stream_configs()

        for i in range(self.num_workers):
            num_streams = stream_configs[i] if i < len(stream_configs) else self.streams_per_worker

            worker = WorkerThread(
                executor=TensorRTExecutor.from_deserialized(self.deserialized_model),
                device=self.device,
                worker_id=i,
                enable_nvtx=self.enable_nvtx,
                num_streams=num_streams,
                asynchronous=self.asynchronous,
                use_graph=self.use_graph
            )
            self.workers.append(worker)
            self.worker_rotation.append(worker)

        for worker in self.workers:
            worker.start()

        total_streams = sum(worker.num_streams for worker in self.workers)
        self.logger.info(f"TensorRT pool with {self.num_workers} workers and {total_streams} total streams ready")

    def _get_stream_configs(self) -> List[int]:
        if self.mixed_stream_config:
            if len(self.mixed_stream_config) != self.num_workers:
                self.logger.warning(
                    f"Mixed stream config length ({len(self.mixed_stream_config)}) doesn't match number of "
                    f"workers ({self.num_workers}). Using default."
                )
                return [self.streams_per_worker] * self.num_workers
            return self.mixed_stream_config
        else:
            return [self.streams_per_worker] * self.num_workers

    def _get_next_worker(self) -> WorkerThread:
        with self.lock:
            worker = self.worker_rotation[0]
            self.worker_rotation.rotate(-1)
            return worker

    @nvtx.annotate("submit_task_to_pool")
    def submit_task(self, input_feed: Dict[str, torch.Tensor]) -> tuple:
        worker = self._get_next_worker()
        task_id = worker.submit(input_feed)
        with self.lock:
            self.total_submitted += 1
        return worker, task_id

    @nvtx.annotate("submit_batch_tasks")
    def submit_batch(self, batch_inputs: List[Dict[str, torch.Tensor]]) -> List[tuple]:
        results = []
        for input_feed in batch_inputs:
            worker, task_id = self.submit_task(input_feed)
            results.append((worker, task_id))
        return results

    @nvtx.annotate("get_batch_results")
    def get_results(self, worker_task_pairs: List[tuple], timeout: float = 10.0) -> List:
        results = []
        pending_pairs = worker_task_pairs.copy()
        start_time = time.time()

        while pending_pairs and (time.time() - start_time < timeout):
            remaining_pairs = []

            for worker, task_id in pending_pairs:
                result = worker.get_result(timeout=5e-4)
                if result and result[0] == task_id:
                    results.append(result[1])
                    with self.lock:
                        self.total_completed += 1
                else:
                    remaining_pairs.append((worker, task_id))

            pending_pairs = remaining_pairs

            if pending_pairs:
                time.sleep(0.001)

        results.extend([None] * len(pending_pairs))

        return results

    def wait_all_completion(self, timeout: float = 30.0) -> bool:
        start_time = time.time()
        while any(worker.has_pending_tasks() for worker in self.workers):
            if time.time() - start_time > timeout:
                self.logger.warning("Timeout waiting for tasks completion")
                return False
            time.sleep(0.01)
        return True

    @nvtx.annotate("stop_worker_pool")
    def stop_all(self):
        self.logger.info("Stopping all workers...")

        self.wait_all_completion(timeout=5.0)

        for worker in self.workers:
            worker.stop()

        self.workers.clear()
        self.worker_rotation.clear()

        self.logger.info("All workers stopped")

    def resize_pool(self, new_size: int, new_streams_per_worker: int = None):
        with self.lock:
            if new_size == self.num_workers and new_streams_per_worker is None:
                return

            with nvtx.annotate(f"resize_pool_from_{self.num_workers}_to_{new_size}"):
                self.logger.info(f"Resizing pool from {self.num_workers} to {new_size} workers")

                if new_streams_per_worker is not None:
                    self.streams_per_worker = new_streams_per_worker

                if new_size > self.num_workers:
                    for i in range(self.num_workers, new_size):
                        worker = WorkerThread(
                            executor=TensorRTExecutor.from_deserialized(self.deserialized_model),
                            device=self.device,
                            worker_id=i,
                            enable_nvtx=self.enable_nvtx,
                            num_streams=self.streams_per_worker,
                            asynchronous=self.asynchronous
                        )
                        worker.start()
                        self.workers.append(worker)
                        self.worker_rotation.append(worker)

                elif new_size < self.num_workers:
                    workers_to_stop = self.workers[new_size:]
                    for worker in workers_to_stop:
                        worker.stop()

                    self.workers = self.workers[:new_size]
                    self.worker_rotation = deque(w for w in self.worker_rotation if w in self.workers)

                self.num_workers = new_size

                total_streams = sum(worker.num_streams for worker in self.workers)
                self.logger.info(f"Pool resized to {new_size} workers with {total_streams} total streams")

    def get_pool_status(self) -> DictConfig:
        workers_stats = [worker.get_stats() for worker in self.workers]

        total_processed = sum(stats["processed_tasks"] for stats in workers_stats)
        total_failed = sum(stats["failed_tasks"] for stats in workers_stats)
        total_pending = sum(stats["pending_tasks"] for stats in workers_stats)
        total_streams = sum(stats["num_streams"] for stats in workers_stats)

        uptime = time.time() - self.start_time
        throughput = total_processed / uptime if uptime > 0 else 0

        return OmegaConf.create({
            "total_workers": self.num_workers,
            "total_streams": total_streams,
            "streams_per_worker": self.streams_per_worker,
            "alive_workers": sum(1 for w in self.workers if w.is_alive()),
            "total_processed_tasks": total_processed,
            "total_failed_tasks": total_failed,
            "total_pending_tasks": total_pending,
            "total_submitted": self.total_submitted,
            "total_completed": self.total_completed,
            "throughput_inferences_per_sec": throughput,
            "uptime_seconds": uptime,
            "workers_status": workers_stats
        })

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_all()
