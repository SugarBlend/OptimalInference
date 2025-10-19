from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path
from typing import List, Literal, Tuple, Union, Dict, Optional

import numpy as np
import tensorrt as trt
import torch
from packaging import version
from pydantic import BaseModel, Field

from deploy2serve.deployment.core.executors.base import BaseExecutor, ExecutorFactory
from deploy2serve.deployment.models.common import Backend
from deploy2serve.utils.logger import get_logger
from wrappers import nvtx_range
import nvtx


class TensorRTAPIAdapter(object):
    def __init__(self, model: trt.ICudaEngine, legacy_mode: bool) -> None:
        self.model: trt.ICudaEngine = model
        self.legacy_mode: bool = legacy_mode

    @property
    def tensor_count(self) -> int:
        return self.model.num_bindings if self.legacy_mode else self.model.num_io_tensors

    def get_name(self, index: int) -> str:
        return self.model.get_binding_name(index) if self.legacy_mode else self.model.get_tensor_name(index)

    def get_dtype(self, index_or_name: Union[str, int]) -> type:
        if self.legacy_mode:
            return trt.nptype(self.model.get_binding_dtype(index_or_name))
        else:
            return trt.nptype(self.model.get_tensor_dtype(index_or_name))

    def get_shape(self, index_or_name: Union[str, int]) -> Tuple[int, ...]:
        if self.legacy_mode:
            return self.model.get_binding_shape(index_or_name)
        else:
            return self.model.get_tensor_shape(index_or_name)

    def is_input(self, index_or_name: Union[str, int]) -> bool:
        if self.legacy_mode:
            return self.model.binding_is_input(index_or_name)
        else:
            return self.model.get_tensor_mode(index_or_name) == trt.TensorIOMode.INPUT


class Binding(BaseModel):
    name: str = Field(description="Node name.")
    dtype: type = Field(description="Type of node tensor.")
    shape: Union[Tuple[int, ...], List[int]] = Field(description="Shape of node tensor.")
    data: torch.Tensor = Field(description="Pytorch tensor pinned for current name of node.")
    ptr: int = Field(description="Address of current named tensor on gpu.")
    io_mode: Literal["output", "input"] = Field(description="Type of node (input / output).")

    def __init__(
        self,
        name: str,
        dtype: type,
        shape: Union[List[int], Tuple[int, ...]],
        data: torch.Tensor,
        ptr: int,
        io_mode: str,
    ) -> None:
        super().__init__(name=name, dtype=dtype, shape=shape, data=data, ptr=ptr, io_mode=io_mode)

    class Config:
        arbitrary_types_allowed = True


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
            self.context = self.get_context()
            self._initialize_io_nodes()
        else:
            self.logger.info("There are no weights available, loading is expected to be done separately.")

        if not torch.cuda.is_available():
            raise RuntimeError("the device must be a derivative of 'cuda', but torch is cpu compiled.")

        self.device: str = device
        self.enable_nvtx: bool = enable_nvtx

        self.cuda_stream = torch.cuda.Stream(device=device)
        self.bindings: OrderedDict[str, Binding] = OrderedDict()
        self.binding_address: OrderedDict[str, int] = OrderedDict()
        self.input_nodes: List[str] = []
        self.output_nodes: List[str] = []
        self.nvtx_marker = nvtx_range if enable_nvtx else nullcontext
        self.num_inferences = 0
        self._last_input_shapes: Dict[str, Tuple[int, ...]] = {}
        self._refresh_addresses = True
        self._cached_addresses: List[int] = []

    @classmethod
    def from_deserialized(cls, model: trt.ICudaEngine, device: str = "cuda:0") -> "TensorRTExecutor":
        instance = cls()
        instance.model = model
        instance.context = instance.get_context()
        instance._initialize_io_nodes()
        return instance

    @nvtx_range()
    def _initialize_io_nodes(self) -> None:
        trt_version = version.parse(trt.__version__)

        if version.parse("8.2.5.1") <= trt_version <= version.parse("8.6.1"):
            adapter = TensorRTAPIAdapter(self.model, legacy_mode=True)
        elif trt_version >= version.parse("9.1.0"):
            adapter = TensorRTAPIAdapter(self.model, legacy_mode=False)
        else:
            raise NotImplementedError(f"TensorRT version {trt.__version__} not supported")

        for index in range(adapter.tensor_count):
            name = adapter.get_name(index)
            is_input = adapter.is_input(index if adapter.legacy_mode else name)

            if is_input:
                self.input_nodes.append(name)
            else:
                self.output_nodes.append(name)

    @staticmethod
    @nvtx_range()
    def _make_binding(name: str, dtype: type, shape: List[int], io_mode: str, device: str) -> Binding:
        tensor = torch.zeros(shape, dtype=getattr(torch, dtype.__name__), device=device)
        return Binding(name=name, dtype=dtype, shape=shape, data=tensor, ptr=int(tensor.data_ptr()), io_mode=io_mode)

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
        if shapes == self._last_input_shapes:
            return

        self._last_input_shapes = shapes.copy()
        trt_version = version.parse(trt.__version__)

        if version.parse("8.2.5.1") <= trt_version <= version.parse("8.6.1"):
            adapter = TensorRTAPIAdapter(self.model, legacy_mode=True)
        elif trt_version >= version.parse("9.1.0"):
            adapter = TensorRTAPIAdapter(self.model, legacy_mode=False)
        else:
            raise NotImplementedError(f"Your version of TensorRT: {trt.__version__} is not implemented")

        for index in range(adapter.tensor_count):
            name = adapter.get_name(index)
            if name in shapes:
                if self.bindings.get(name):
                    recreate = shapes[name] != self.bindings[name].shape
                else:
                    recreate = True

                if recreate:
                    dtype = adapter.get_dtype(index if adapter.legacy_mode else name)
                    shape = shapes.get(name)
                    if shape is None:
                        shape = adapter.get_shape(index if adapter.legacy_mode else name)
                    io_mode = "input" if adapter.is_input(index if adapter.legacy_mode else name) else "output"

                    self.bindings[name] = TensorRTExecutor._make_binding(
                        name, dtype, list(shape), io_mode, self.device
                    )
                    self._refresh_addresses = True

    @nvtx_range()
    def _get_binding_index(self, name: str) -> Optional[int]:
        trt_version = version.parse(trt.__version__)
        if version.parse("8.2.5.1") <= trt_version <= version.parse("8.6.1"):
            for idx in range(self.model.num_bindings):
                if self.model.get_binding_name(idx) == name:
                    return idx
        return None

    @nvtx_range()
    def _execute_async(self, is_new_api: bool) -> None:
        if is_new_api:
            if self._refresh_addresses:
                for node in self.bindings:
                    self.context.set_tensor_address(node, self.binding_address[node])
                self._refresh_addresses = False
            self.context.execute_async_v3(self.cuda_stream.cuda_stream)
        else:
            if not hasattr(self, "_cached_addresses") or self._refresh_addresses:
                self._cached_addresses = [
                    self.binding_address.get(self.model.get_binding_name(i), 0)
                    for i in range(self.model.num_bindings)
                ]
                self._refresh_addresses = False

            self.context.execute_async_v2(
                bindings=self._cached_addresses,
                stream_handle=self.cuda_stream.cuda_stream)

    @nvtx_range()
    def _get_tensor_dtype(self, name: str) -> type:
        trt_version = version.parse(trt.__version__)

        if version.parse("8.2.5.1") <= trt_version <= version.parse("8.6.1"):
            adapter = TensorRTAPIAdapter(self.model, legacy_mode=True)
            idx = self._get_binding_index(name)
            return adapter.get_dtype(idx) if idx is not None else np.float32
        elif trt_version >= version.parse("9.1.0"):
            adapter = TensorRTAPIAdapter(self.model, legacy_mode=False)
            return adapter.get_dtype(name)
        else:
            return np.float32

    @nvtx_range()
    def _prepare_output_bindings(self, is_new_api: bool, input_feed: Dict[str, torch.Tensor]) -> None:
        for output_node in self.output_nodes:
            # TODO: here problem with random output
            # if output_node not in self.bindings:
                shape = self._get_output_shape(output_node, is_new_api)
                if shape and all(dim > 0 for dim in shape):
                    dtype = self._get_tensor_dtype(output_node)
                    self.bindings[output_node] = self._make_binding(
                        output_node, dtype, list(shape), "output", self.device
                    )
                    self.binding_address[output_node] = self.bindings[output_node].ptr

    @nvtx_range()
    def _get_output_shape(self, name: str, is_new_api: bool) -> Tuple[int, ...]:
        if is_new_api:
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
    def infer(self, input_feed: Dict[str, torch.Tensor], asynchronous: bool = False, **kwargs) -> List[torch.Tensor]:
        self._ensure_initialized()

        with torch.cuda.stream(self.cuda_stream):
            with nvtx.annotate(f"update_input_bindings.{self.num_inferences}", color="green"):
                input_shapes = {node: input_feed[node].shape for node in input_feed}
                self.update_bindings(input_shapes)

                is_new_api = version.parse(trt.__version__) >= version.parse("9.1.0")
                for node in input_feed:
                    if (
                        input_feed[node].device != torch.device(self.device) or
                        input_feed[node].dtype != self.bindings[node].data.dtype or
                        not input_feed[node].is_contiguous()
                    ):
                        input_feed[node] = input_feed[node].to(
                            device=self.device,
                            dtype=self.bindings[node].data.dtype,
                            non_blocking=True
                        ).contiguous()
                    self.binding_address[node] = int(input_feed[node].data_ptr())

                    if is_new_api:
                        self.context.set_input_shape(node, input_feed[node].shape)
                    else:
                        binding_index = self._get_binding_index(node)
                        if binding_index is not None:
                            self.context.set_binding_shape(binding_index, input_feed[node].shape)

            with nvtx.annotate(f"prepare_output_bindings.{self.num_inferences}", color="red"):
                self._prepare_output_bindings(is_new_api, input_feed)

            with nvtx.annotate(f"tensorrt_execute.{self.num_inferences}", color="blue"):
                if asynchronous:
                    self._execute_async(is_new_api)
                else:
                    if not hasattr(self, "_cached_addresses") or self._refresh_addresses:
                        trt_version = version.parse(trt.__version__)
                        if version.parse("8.2.5.1") <= trt_version <= version.parse("8.6.1"):
                            adapter = TensorRTAPIAdapter(self.model, legacy_mode=True)
                        elif trt_version >= version.parse("9.1.0"):
                            adapter = TensorRTAPIAdapter(self.model, legacy_mode=False)

                        self._cached_addresses = [
                            self.binding_address.get(adapter.get_name(i))
                            for i in range(adapter.tensor_count)
                        ]
                        self._refresh_addresses = False
                    self.context.execute_v2(list(self.binding_address.values()))

            self.num_inferences += 1
        return [self.bindings[node].data for node in self.output_nodes]
