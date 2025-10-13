from functools import wraps
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