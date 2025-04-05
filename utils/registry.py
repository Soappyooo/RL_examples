from typing import Callable, Optional

ENV_REGISTRY: dict = {}
AGENT_REGISTRY: dict = {}


def register_env(name: Optional[str] = None) -> Callable:
    def decorator(cls: type) -> type:
        nonlocal name
        if name is None:
            name = cls.__name__
        ENV_REGISTRY[name] = cls
        return cls

    return decorator


def register_agent(name: Optional[str] = None) -> Callable:
    def decorator(cls: type) -> type:
        nonlocal name
        if name is None:
            name = cls.__name__
        AGENT_REGISTRY[name] = cls
        return cls

    return decorator
