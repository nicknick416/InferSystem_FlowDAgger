"""可注册基类 Mixin — 消除注册器模式的重复代码。

提供 @register() 装饰器和 from_config() 工厂方法的标准实现。

用法::

    class BaseCamera(Registrable["BaseCamera"]):
        _registry: ClassVar[dict[str, type[BaseCamera]]] = {}
        _registry_label: ClassVar[str] = "相机"

        @classmethod
        def _from_config_dict(cls, name, cfg, **kw):
            raise NotImplementedError

    @BaseCamera.register("realsense")
    class RealSenseCamera(BaseCamera):
        ...
"""
from __future__ import annotations

import logging
from typing import Any, ClassVar, Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="Registrable")


class Registrable(Generic[T]):
    """注册器 Mixin。

    子类需要声明两个类变量:
      _registry: dict[str, type[T]] = {}     注册表
      _registry_label: str = "组件"           日志用的中文名

    提供:
      register(type_name)   装饰器，将子类注册到 _registry
    """

    _registry: ClassVar[dict[str, type]] = {}
    _registry_label: ClassVar[str] = "组件"

    @classmethod
    def register(cls, type_name: str):
        """装饰器: 将子类注册到工厂表。"""
        def decorator(subcls: type[T]) -> type[T]:
            if type_name in cls._registry:
                logger.warning(
                    "%s类型 '%s' 已注册为 %s，将被 %s 覆盖",
                    cls._registry_label,
                    type_name,
                    cls._registry[type_name].__name__,
                    subcls.__name__,
                )
            cls._registry[type_name] = subcls
            return subcls
        return decorator

    @classmethod
    def _resolve_factory(cls, type_name: str) -> type[T]:
        """根据类型名查找注册的子类，未找到则抛出 ValueError。"""
        factory = cls._registry.get(type_name)
        if factory is None:
            available = ", ".join(sorted(cls._registry.keys()))
            raise ValueError(
                f"未知的{cls._registry_label}类型 '{type_name}'。"
                f"可用: {available}"
            )
        return factory
