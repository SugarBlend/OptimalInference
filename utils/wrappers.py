from functools import wraps
import torch
from typing import TypeVar, Callable, Any, cast, Optional
import nvtx

F = TypeVar("F", bound=Callable[..., Any])


def nvtx_range(func: Optional[F] = None, name: Optional[str] = None) -> Callable[[F], F] | F:
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            range_name = name or func.__qualname__
            nvtx.push_range(range_name, color="gray")
            try:
                return func(*args, **kwargs)
            finally:
                nvtx.pop_range()

        return cast(F, wrapper)

    if func is None:
        return decorator
    return decorator(func)


def memory_snap(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if torch.cuda.is_available():
            # torch.cuda.synchronize()
            start_allocated = torch.cuda.memory_allocated()
            start_cached = torch.cuda.memory_reserved()

        result = func(*args, **kwargs)

        if torch.cuda.is_available():
            # torch.cuda.synchronize()
            end_allocated = torch.cuda.memory_allocated()
            end_cached = torch.cuda.memory_reserved()

            allocated_diff = (end_allocated - start_allocated) / 1024 ** 2
            cached_diff = (end_cached - start_cached) / 1024 ** 2

            print(f"{func.__name__} - Memory change: Allocated: {allocated_diff:.2f} MB, Cached: {cached_diff:.2f} MB")

        return result

    return wrapper
