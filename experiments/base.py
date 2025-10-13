from deploy2serve.deployment.models.common import LoggingMeta
from queue import Queue, Empty
import time
import torch
from threading import Thread, Lock
from typing import Dict, List, Optional, Tuple
from omegaconf import OmegaConf, DictConfig
from override import TensorRTExecutor
import threading
import nvtx


class TensorRTWorkerThread(Thread, metaclass=LoggingMeta):
    def __init__(
        self,
        executor: TensorRTExecutor,
        device: str = "cuda:0",
        worker_id: int = 0,
        daemon: bool = True,
        enable_nvtx: bool = True
    ):
        super().__init__(
            name=f"TRT-Worker-{worker_id}",
            daemon=daemon
        )
        self.completion_events: Dict[int, torch.cuda.Event] = {}
        self.event_lock = threading.Lock()

        self.executor: TensorRTExecutor = executor
        self.worker_id: int = worker_id
        self.device: str = device
        self.enable_nvtx: bool = enable_nvtx

        self.input_queue: Queue[int, Dict[str, torch.Tensor]] = Queue()
        self.result_queue: Queue[int, List[torch.Tensor]] = Queue()
        self._running: bool = False

        self.processed_tasks: int = 0
        self.failed_tasks: int = 0
        self.lock: Lock = Lock()

    @property
    def running(self) -> bool:
        return self._running and self.is_alive()

    def submit(self, input_feed: Dict[str, torch.Tensor]) -> int:
        task_id = int(time.time() * 1000) + self.worker_id * 10000 + self.processed_tasks

        # Create synchronization marker for current task_id
        completion_event = torch.cuda.Event(enable_timing=False)
        with self.event_lock:
            self.completion_events[task_id] = completion_event

        self.input_queue.put((task_id, input_feed))
        return task_id

    def get_completion_event(self, task_id: int) -> Optional[torch.cuda.Event]:
        with self.event_lock:
            return self.completion_events.get(task_id)

    def get_result(self, timeout: float = None) -> Optional[Tuple]:
        try:
            return self.result_queue.get(timeout=timeout)
        except Empty:
            return None

    @nvtx.annotate(f"_process_next_task")
    def _process_next_task(self) -> None:
        def push_results(task_id: int, results: Optional[List[torch.Tensor]] = None) -> None:
            # Record point to get synchronized results from parallel tasks
            with self.event_lock:
                event = self.completion_events.get(task_id)
                if event is not None:
                    event.record(self.executor.cuda_stream)
            self.result_queue.put((task_id, results))

        try:
            task_id, input_feed = self.input_queue.get(timeout=0.01)
            with nvtx.annotate(f"Worker_{self.worker_id}_process_task"):
                results = self._execute_inference(input_feed)
                push_results(task_id, results)
                with self.lock:
                    self.processed_tasks += 1
        except Empty:
            time.sleep(1e-4)
        except Exception as error:
            with self.lock:
                self.failed_tasks += 1
            self.logger.error(f"Worker error: {error}")
            push_results(task_id)

    def cleanup_completion_event(self, task_id: int) -> None:
        with self.event_lock:
            self.completion_events.pop(task_id, None)

    def has_pending_tasks(self) -> bool:
        return not self.input_queue.empty()

    def pending_tasks_count(self) -> int:
        return self.input_queue.qsize()

    def get_stats(self) -> DictConfig:
        with self.lock:
            stats_dict = {
                "worker_id": self.worker_id,
                "processed_tasks": self.processed_tasks,
                "failed_tasks": self.failed_tasks,
                "pending_tasks": self.pending_tasks_count(),
                "is_alive": self.is_alive()
            }
        return OmegaConf.create(stats_dict)

    def stop(self) -> None:
        self._running = False
        if self.is_alive():
            self.join(timeout=5.0)
            if self.is_alive():
                self.logger.warning(f"Worker {self.worker_id} didn't stop gracefully")
            else:
                self.logger.info(f"Worker {self.worker_id} stopped")

    def run(self) -> None:
        self._running = True
        nvtx.mark(f"Worker_{self.worker_id}_started")
        self.logger.info(f"Worker {self.worker_id} started on stream {self.executor.cuda_stream.cuda_stream}")

        try:
            while self._running:
                self._process_next_task()
        except Exception as error:
            self.logger.critical(f"Worker {self.worker_id} crashed: {error}")
        finally:
            self._running = False
            nvtx.mark(f"Worker_{self.worker_id}_finished")
            self.logger.info(f"Worker {self.worker_id} finished")

    def _execute_inference(self, input_feed: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        with nvtx.annotate(f"Worker_{self.worker_id}_inference"):
            try:
                results = self.executor.infer(input_feed, asynchronous=False)
                return results
            except Exception as error:
                self.logger.error(f"Inference error in worker {self.worker_id}: {error}")
                raise

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
