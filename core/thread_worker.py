from contextlib import nullcontext
from collections import deque
from deploy2serve.deployment.models.common import LoggingMeta
from omegaconf import OmegaConf, DictConfig
import nvtx
from queue import Queue, Empty
import time
import torch
from threading import Thread, Lock, Barrier
from typing import Dict, List, Optional, Tuple, Union
import threading

from core.override import BaseExecutor


class WorkerThread(Thread, metaclass=LoggingMeta):
    def __init__(
        self,
        executor: BaseExecutor,
        device: str = "cuda:0",
        worker_id: int = 0,
        daemon: bool = True,
        enable_nvtx: bool = True,
        num_streams: int = 1,
        asynchronous: bool = False,
        use_graph: bool = False,
        capture_barrier: Optional[Barrier] = None,
        mutex: Union[nullcontext, Lock] = nullcontext(),
        use_unique_context: bool = False
    ):
        super().__init__(name=f"WorkerThread-{worker_id}", daemon=daemon)
        self.completion_events: Dict[int, torch.cuda.Event] = {}
        self.event_lock = threading.Lock()

        self.executor: BaseExecutor = executor
        self.worker_id: int = worker_id
        self.device: str = device
        self.enable_nvtx: bool = enable_nvtx
        self.num_streams: int = num_streams
        self.asynchronous: bool = asynchronous
        self.use_graph: bool = use_graph
        self.capture_barrier: Optional[Barrier] = capture_barrier
        self.mutex: Union[nullcontext, Lock] = mutex
        self.use_unique_context: bool = use_unique_context

        self.streams: List[torch.cuda.Stream] = []
        self._initialize_streams()

        self.stream_rotation = deque(range(self.num_streams))
        self.stream_lock = threading.Lock()

        self.input_queue: Queue[Tuple[int, Dict[str, torch.Tensor], int]] = Queue()
        self.result_queue: Queue[Tuple[int, List[torch.Tensor]]] = Queue()
        self._running: bool = False

        self.processed_tasks: int = 0
        self.failed_tasks: int = 0
        self.lock: Lock = Lock()

    @property
    def running(self) -> bool:
        return self._running and self.is_alive()

    @property
    def has_pending_tasks(self) -> bool:
        return not self.input_queue.empty()

    @property
    def pending_tasks_count(self) -> int:
        return self.input_queue.qsize()

    def _initialize_streams(self) -> None:
        for i in range(self.num_streams):
            stream = torch.cuda.Stream(device=self.device)
            self.streams.append(stream)
        self.logger.info(f"Worker {self.worker_id} initialized with {self.num_streams} CUDA streams")

    def _get_next_stream(self) -> Tuple[int, torch.cuda.Stream]:
        with self.stream_lock:
            stream_id = self.stream_rotation[0]
            self.stream_rotation.rotate(-1)
            return stream_id, self.streams[stream_id]

    def submit(self, input_feed: Dict[str, torch.Tensor]) -> int:
        task_id = int(time.time() * 1000) + self.worker_id * 10000 + self.processed_tasks

        stream_id, stream = self._get_next_stream()

        completion_event = torch.cuda.Event(enable_timing=False)
        with self.event_lock:
            self.completion_events[task_id] = completion_event

        self.input_queue.put((task_id, input_feed, stream_id))
        return task_id

    def get_completion_event(self, task_id: int) -> Optional[torch.cuda.Event]:
        with self.event_lock:
            return self.completion_events.get(task_id)

    def get_result(self, timeout: float = None) -> Optional[Tuple]:
        try:
            return self.result_queue.get(timeout=timeout)
        except Empty:
            return None

    def cleanup_completion_event(self, task_id: int) -> None:
        with self.event_lock:
            self.completion_events.pop(task_id, None)

    def get_stats(self) -> DictConfig:
        with self.lock:
            stats_dict = {
                "worker_id": self.worker_id,
                "num_streams": self.num_streams,
                "processed_tasks": self.processed_tasks,
                "failed_tasks": self.failed_tasks,
                "pending_tasks": self.pending_tasks_count,
                "is_alive": self.is_alive()
            }
        return OmegaConf.create(stats_dict)

    def stop(self) -> None:
        self._running = False

        for stream in self.streams:
            stream.synchronize()

        if self.is_alive():
            self.join(timeout=5.0)
            if self.is_alive():
                self.logger.warning(f"Worker {self.worker_id} didn't stop gracefully")
            else:
                self.logger.info(f"Worker {self.worker_id} stopped")

    def run(self) -> None:
        self._running = True
        nvtx.mark(f"Worker_{self.worker_id}_started")
        self.logger.info(f"Worker {self.worker_id} started with {self.num_streams} streams")

        try:
            while self._running:
                self._process_next_task()
        except Exception as error:
            self.logger.critical(f"Worker {self.worker_id} crashed: {error}")
        finally:
            self._running = False
            nvtx.mark(f"Worker_{self.worker_id}_finished")
            self.logger.info(f"Worker {self.worker_id} finished")

    @nvtx.annotate("_process_next_task")
    def _process_next_task(self) -> None:
        def push_results(task_id: int, results: Optional[List[torch.Tensor]] = None) -> None:
            # Record point to get synchronized results from parallel tasks
            with self.event_lock:
                event = self.completion_events.get(task_id)
                if event is not None:
                    event.record(self.streams[stream_id])
            self.result_queue.put_nowait((task_id, results))

        try:
            task_id, input_feed, stream_id = self.input_queue.get_nowait()

            with nvtx.annotate(f"Worker:{self.worker_id} stream:{stream_id} process task:{task_id}"):
                results = self._execute_inference(input_feed, self.streams[stream_id])
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

    def _execute_inference(self, input_feed: Dict[str, torch.Tensor], stream: torch.cuda.Stream) -> List[torch.Tensor]:
        stream_name = f"Worker_{self.worker_id}_stream_{self.streams.index(stream)}"
        with nvtx.annotate(f"{stream_name}_inference"):
            try:
                self.executor.cuda_stream = stream
                results = self.executor.infer(input_feed=input_feed, asynchronous=self.asynchronous,
                                              use_graph=self.use_graph, capture_barrier=self.capture_barrier,
                                              use_unique_context=self.use_unique_context, mutex=self.mutex)
                return results
            except Exception as error:
                self.logger.error(f"Inference error in {stream_name}: {error}")
                raise

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
