from typing import List, Tuple, Union, Any
import tensorrt as trt
import torch
from pydantic import BaseModel, Field


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


class TensorMetadata(BaseModel):
    name: str = Field(description="Name of node in tensorrt graph.")
    dtype: Any = Field(description="Data type for current node name.")
    shape: Union[Tuple[int, ...], List[int]] = Field(description="Current static shape for this node.")
    is_input: bool = Field(description="Is the node an input node?")

    def __init__(
        self,
        name: str,
        dtype: Any,
        shape: Tuple[int, ...],
        is_input: bool,
    ) -> None:
        dtype = getattr(torch, dtype.__name__)
        super().__init__(name=name, dtype=dtype, shape=shape, is_input=is_input)

    class Config:
        arbitrary_types_allowed = True


class TensorBinding(object):
    def __init__(self, metadata: TensorMetadata, tensor: torch.Tensor):
        self.metadata: TensorMetadata = metadata
        self.tensor: torch.Tensor = tensor

    @property
    def dtype(self) -> torch.dtype:
        return self.tensor.dtype

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.tensor.shape)

    @property
    def ptr(self) -> int:
        return int(self.tensor.data_ptr())

    @classmethod
    def create(cls, metadata: TensorMetadata, device: str) -> 'TensorBinding':
        tensor = torch.zeros(metadata.shape, dtype=metadata.dtype, device=device)
        return cls(metadata, tensor)
