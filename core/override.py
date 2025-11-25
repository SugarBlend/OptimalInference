from collections import OrderedDict
from contextlib import nullcontext
from packaging import version
from pathlib import Path
from typing import List, Tuple, Union, Dict, Optional
import tensorrt as trt
import torch
from threading import Lock, Barrier

from deploy2serve.deployment.core.executors.base import BaseExecutor, ExecutorFactory
from deploy2serve.deployment.models.common import Backend
from deploy2serve.utils.logger import get_logger
from utils.wrappers import nvtx_range
import nvtx
from utils.bindings import TensorBinding, TensorMetadata, TensorRTAPIAdapter


class LoggingMixin:
    @property
    def logger(self):
        if not hasattr(self, '_logger'):
            self._logger = get_logger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        return self._logger


@ExecutorFactory.register(Backend.TensorRT)
class TensorRTExecutor(BaseExecutor, LoggingMixin):
    model: trt.ICudaEngine
    context: trt.IExecutionContext
    cuda_stream: torch.cuda.Stream

    def __init__(
        self,
        checkpoints_path: Optional[str] = None,
        device: str = "cuda:0",
        enable_nvtx: bool = True
    ) -> None:
        if not(checkpoints_path is None):
            if not Path(self.checkpoints_path).is_absolute():
                self.checkpoints_path = Path.cwd().joinpath(self.checkpoints_path).as_posix()
            else:
                self.checkpoints_path: str = checkpoints_path
            self.model = self.load(checkpoints_path, device)
            self._initialize_io_nodes()
        else:
            self.logger.info("There are no weights available, loading is expected to be done separately.")

        if not torch.cuda.is_available():
            raise RuntimeError("the device must be a derivative of 'cuda', but torch is cpu compiled.")

        self.device: str = device
        self.enable_nvtx: bool = enable_nvtx

        self.trt_version = version.parse(trt.__version__)
        self.is_new_api = self.trt_version >= version.parse("9.1.0")

        # dependency on cuda stream
        self.cuda_stream = torch.cuda.Stream(device=device)
        self.cuda_graphs: Dict[str, torch.cuda.CUDAGraph] = {}
        self.bindings: OrderedDict[int, OrderedDict[str, TensorBinding]] = OrderedDict()
        self.num_inferences: Dict[int, int] = {}
        self.input_tensors: Dict[str, Dict[str, torch.Tensor]] = {}
        #TODO: Extend to use in case when have got several optimization profiles
        self.contexts: Dict[int, trt.IExecutionContext] = {}
        self.addresses: List[int] = []

        self.input_nodes: List[str] = []
        self.output_nodes: List[str] = []
        self.nvtx_marker = nvtx_range if enable_nvtx else nullcontext
        self._last_input_shapes: Dict[str, Tuple[int, ...]] = {}

        self.context: Optional[trt.IExecutionContext] = None

    @classmethod
    def from_deserialized(cls, model: trt.ICudaEngine, device: str = "cuda:0") -> "TensorRTExecutor":
        instance = cls()
        instance.model = model
        instance._initialize_io_nodes()
        return instance

    @nvtx_range()
    def _initialize_io_nodes(self) -> None:
        if version.parse("8.2.5.1") <= self.trt_version <= version.parse("8.6.1"):
            self.adapter = TensorRTAPIAdapter(self.model, legacy_mode=True)
        elif self.is_new_api:
            self.adapter = TensorRTAPIAdapter(self.model, legacy_mode=False)
        else:
            raise NotImplementedError(f"TensorRT version {trt.__version__} not supported")

        for index in range(self.adapter.tensor_count):
            name = self.adapter.get_name(index)
            is_input = self.adapter.is_input(index if self.adapter.legacy_mode else name)

            if is_input:
                self.input_nodes.append(name)
            else:
                self.output_nodes.append(name)

    @staticmethod
    @nvtx_range()
    def load(
        weights_path: Union[str, Path],
        device: str,
        log_level: trt.Logger.Severity = trt.Logger.Severity.ERROR
    ) -> trt.ICudaEngine:
        path = Path(weights_path)
        if not path.exists():
            raise FileNotFoundError(f"TensorRT model file not found at: '{path}'.")

        logger = trt.Logger()
        trt.init_libnvinfer_plugins(logger, namespace="")
        with path.open("rb") as file, trt.Runtime(logger) as runtime:
            model = runtime.deserialize_cuda_engine(file.read())
        return model

    @nvtx_range()
    def get_context(self) -> trt.IExecutionContext:
        return self.model.create_execution_context()

    @nvtx_range()
    def update_bindings(self, shapes: Dict[str, Tuple[int, ...]]) -> None:
        if shapes == self._last_input_shapes.get(self.cuda_stream.cuda_stream):
            return

        self._last_input_shapes[self.cuda_stream.cuda_stream] = shapes.copy()

        if not self.bindings.get(self.cuda_stream.cuda_stream):
            self.bindings[self.cuda_stream.cuda_stream] = OrderedDict()

        for index in range(self.adapter.tensor_count):
            name = self.adapter.get_name(index)
            if name in shapes:
                if self.bindings[self.cuda_stream.cuda_stream].get(name):
                    recreate = shapes[name] != self.bindings[self.cuda_stream.cuda_stream][name].tensor.shape
                else:
                    recreate = True

                if recreate:
                    node = index if self.adapter.legacy_mode else name
                    dtype = self.adapter.get_dtype(node)
                    shape = shapes.get(name)
                    if shape is None:
                        shape = self.adapter.get_shape(node)
                    is_input = self.adapter.is_input(node)

                    metadata = TensorMetadata(name, dtype, list(shape), is_input)
                    self.bindings[self.cuda_stream.cuda_stream][name] = TensorBinding.create(
                        metadata, self.device
                    )

                    if is_input:
                        if not self.input_tensors.get(self.cuda_stream.cuda_stream):
                            self.input_tensors[self.cuda_stream.cuda_stream] = {}
                        self.input_tensors[self.cuda_stream.cuda_stream][name] = self.bindings[self.cuda_stream.cuda_stream][name].tensor

                        if self.is_new_api:
                            self.context.set_input_shape(node, self.input_tensors[self.cuda_stream.cuda_stream][name].shape)
                        else:
                            binding_index = self._get_binding_index(node)
                            if binding_index is not None:
                                self.context.set_binding_shape(binding_index, self.input_tensors[self.cuda_stream.cuda_stream][name].shape)

    @nvtx_range()
    def _update_output_bindings(self) -> None:
        for output_node in self.output_nodes:
            #TODO: Need to add checking of dynamic shapes and processed case with negative dimensions
            if output_node not in self.bindings[self.cuda_stream.cuda_stream]:
                shape = self._get_output_shape(output_node)
                if shape and all(dim > 0 for dim in shape):
                    dtype = self._get_node_dtype(output_node)
                    metadata = TensorMetadata(output_node, dtype, list(shape), False)
                    self.bindings[self.cuda_stream.cuda_stream][output_node] = TensorBinding.create(
                        metadata, self.device
                    )

    @nvtx_range()
    def _get_binding_index(self, name: str) -> Optional[int]:
        trt_version = version.parse(trt.__version__)
        if version.parse("8.2.5.1") <= trt_version <= version.parse("8.6.1"):
            for idx in range(self.model.num_bindings):
                if self.model.get_binding_name(idx) == name:
                    return idx
        return None

    @nvtx_range()
    def _get_node_dtype(self, name: str) -> type:
        if version.parse("8.2.5.1") <= self.trt_version <= version.parse("8.6.1"):
            idx = self._get_binding_index(name)
            return self.adapter.get_dtype(idx)
        elif self.is_new_api:
            return self.adapter.get_dtype(name)
        else:
            raise NotImplementedError(f"TensorRT version {trt.__version__} not supported")

    @nvtx_range()
    def _get_output_shape(self, name: str) -> Tuple[int, ...]:
        if self.is_new_api:
            return tuple(self.context.get_tensor_shape(name))
        else:
            idx = self._get_binding_index(name)
            if idx is not None:
                return tuple(self.context.get_binding_shape(idx))
        return tuple()

    def _ensure_initialized(self) -> None:
        if self.model is None or self.context is None:
            raise RuntimeError(
                "TensorRTExecutor not properly initialized. Call from_deserialized() or provide checkpoints_path"
            )

    @nvtx_range()
    @torch.no_grad()
    def infer(
        self,
        input_feed: Dict[str, torch.Tensor],
        asynchronous: bool = False,
        use_graph: bool = False,
        capture_barrier: Optional[Barrier] = None,
        use_unique_context: bool = False,
        mutex: Union[nullcontext, Lock] = nullcontext(),
        **kwargs
    ) -> List[torch.Tensor]:
        if use_unique_context:
            if not self.contexts.get(self.cuda_stream.cuda_stream):
                self.contexts[self.cuda_stream.cuda_stream] = self.get_context()
            self.context = self.contexts[self.cuda_stream.cuda_stream]
        else:
            if self.context is None:
                self.context = self.get_context()

        if not self.num_inferences.get(self.cuda_stream.cuda_stream):
            self.num_inferences[self.cuda_stream.cuda_stream] = 0

        self._ensure_initialized()

        self.cuda_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.cuda_stream):
            with nvtx.annotate(f"update_input_bindings.{self.num_inferences[self.cuda_stream.cuda_stream]}", color="green"):
                input_shapes = {node: input_feed[node].shape for node in input_feed}
                self.update_bindings(input_shapes)

                for node in input_feed:
                    if (
                        input_feed[node].device != torch.device(self.device) or
                        input_feed[node].dtype != self.bindings[self.cuda_stream.cuda_stream][node].dtype or
                        not input_feed[node].is_contiguous()
                    ):
                        input_feed[node] = input_feed[node].to(
                            device=self.device,
                            dtype=self.bindings[self.cuda_stream.cuda_stream][node].dtype,
                            non_blocking=True
                        ).contiguous()
                        self.input_tensors[self.cuda_stream.cuda_stream][node].copy_(input_feed[node])

            with nvtx.annotate(f"update_output_bindings.{self.num_inferences[self.cuda_stream.cuda_stream]}", color="red"):
                self._update_output_bindings()

            with nvtx.annotate(f"tensorrt_execute.{self.num_inferences[self.cuda_stream.cuda_stream]}", color="blue"):
                if asynchronous:
                    self._execute_async(use_graph, capture_barrier, mutex)
                else:
                    self._execute_sync()

            self.num_inferences[self.cuda_stream.cuda_stream] += 1
        torch.cuda.default_stream().wait_stream(self.cuda_stream)
        torch.cuda.current_stream().wait_stream(self.cuda_stream)

        results = [self.bindings[self.cuda_stream.cuda_stream][node].tensor.clone() for node in self.output_nodes]
        return results

    @nvtx_range()
    def _execute_async(self, use_graph: bool, capture_barrier: Barrier, mutex: Union[nullcontext, Lock]) -> None:
        if self.cuda_graphs.get(self.cuda_stream.cuda_stream) and use_graph:
            self.cuda_graphs[self.cuda_stream.cuda_stream].replay()
            return

        if not self.num_inferences[self.cuda_stream.cuda_stream]:
            ret = all(map(lambda node: self.context.set_tensor_address(node, self.bindings[
                self.cuda_stream.cuda_stream][node].ptr), self.bindings[self.cuda_stream.cuda_stream]))
            if not ret:
                raise RuntimeError("Failed to set tensor addresses!")

        if use_graph:
            if not isinstance(capture_barrier, Barrier) or not isinstance(mutex, Lock):
                raise Exception("Graph capture must be isolated and not parallel to other threads.")

            with nvtx.annotate(f"graph capture: {self.cuda_stream.cuda_stream}"):
                capture_barrier.wait()
                with mutex:
                    self.cuda_graphs[self.cuda_stream.cuda_stream] = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(self.cuda_graphs[self.cuda_stream.cuda_stream], stream=self.cuda_stream):
                        self.context.execute_async_v3(self.cuda_stream.cuda_stream)
                capture_barrier.wait()
        self.context.execute_async_v3(self.cuda_stream.cuda_stream)

    @nvtx_range()
    def _execute_sync(self) -> None:
        if not self.num_inferences[self.cuda_stream.cuda_stream]:
            self.addresses = [item.ptr for item in self.bindings[self.cuda_stream.cuda_stream].values()]

        torch.cuda.default_stream().wait_stream(self.cuda_stream)
        with torch.cuda.stream(self.cuda_stream):
            self.context.execute_v2(self.addresses)
