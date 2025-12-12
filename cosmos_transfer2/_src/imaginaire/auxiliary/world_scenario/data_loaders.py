# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

"""
Data loader interface and registration system for different data formats.

This module provides an abstract interface for loading scene data from various
sources (MADS ClipGT, RQS-HQ webdataset, etc.) into the unified SceneData representation.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

from loguru import logger

from cosmos_transfer2._src.imaginaire.auxiliary.world_scenario.data_types import SceneData


class SceneDataLoader(ABC):
    """Abstract base class for scene data loaders."""

    @abstractmethod
    def can_load(self, source: Union[Path, str, Dict[str, Any]]) -> bool:
        """
        Check if this loader can handle the given source.

        Args:
            source: Data source (path, URL, config dict, etc.)

        Returns:
            True if this loader can handle the source
        """
        pass

    @abstractmethod
    def load(
        self,
        source: Union[Path, str, Dict[str, Any]],
        camera_names: Optional[List[str]] = None,
        max_frames: int = -1,
        **kwargs: Any,
    ) -> SceneData:
        """
        Load scene data from the source.

        Args:
            source: Data source
            camera_names: Optional list of camera names to load
            max_frames: Maximum number of frames to load (-1 for all)
            **kwargs: Additional loader-specific arguments

        Returns:
            Loaded scene data
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the loader name for registration."""
        pass

    @property
    def description(self) -> str:
        """Get a description of what this loader handles."""
        return f"Loader for {self.name} format"


class SceneDataLoaderRegistry:
    """Registry for scene data loaders."""

    def __init__(self) -> None:
        """Initialize the loader registry."""
        self._loaders: Dict[str, SceneDataLoader] = {}
        self._loader_order: List[str] = []  # For priority ordering

    def register(
        self,
        loader: SceneDataLoader,
        priority: int = 0,
    ) -> None:
        """
        Register a data loader.

        Args:
            loader: The loader instance to register
            priority: Priority for auto-detection (higher = checked first)
        """
        name = loader.name
        if name in self._loaders:
            logger.warning(f"Overwriting existing loader: {name}")

        self._loaders[name] = loader

        # Insert in priority order
        self._loader_order = sorted(
            set([*self._loader_order, name]),
            key=lambda n: -priority if n == name else self._loader_order.index(n) if n in self._loader_order else 0,
        )

        logger.debug(f"Registered loader: {name} (priority={priority})")

    def unregister(self, name: str) -> None:
        """Unregister a loader by name."""
        if name in self._loaders:
            del self._loaders[name]
            self._loader_order.remove(name)
            logger.debug(f"Unregistered loader: {name}")

    def get_loader(self, name: str) -> Optional[SceneDataLoader]:
        """Get a specific loader by name."""
        return self._loaders.get(name)

    def find_loader(self, source: Union[Path, str, Dict[str, Any]]) -> Optional[SceneDataLoader]:
        """
        Find a suitable loader for the given source.

        Args:
            source: Data source

        Returns:
            First loader that can handle the source, or None
        """
        for name in self._loader_order:
            loader = self._loaders[name]
            if loader.can_load(source):
                logger.debug(f"Found suitable loader: {name}")
                return loader
        return None

    def load(
        self,
        source: Union[Path, str, Dict[str, Any]],
        loader_name: Optional[str] = None,
        **kwargs: Any,
    ) -> SceneData:
        """
        Load scene data using an appropriate loader.

        Args:
            source: Data source
            loader_name: Optional specific loader to use
            **kwargs: Additional arguments passed to loader

        Returns:
            Loaded scene data

        Raises:
            ValueError: If no suitable loader is found
        """
        if loader_name:
            loader = self.get_loader(loader_name)
            if not loader:
                raise ValueError(f"Loader not found: {loader_name}")
        else:
            loader = self.find_loader(source)
            if not loader:
                available = ", ".join(self._loaders.keys())
                raise ValueError(f"No suitable loader found for source. Available: {available}")

        logger.debug(f"Loading with {loader.name}: {source}")
        return loader.load(source, **kwargs)

    def list_loaders(self) -> List[str]:
        """Get list of registered loader names in priority order."""
        return self._loader_order.copy()

    def describe_loaders(self) -> Dict[str, str]:
        """Get descriptions of all registered loaders."""
        return {name: loader.description for name, loader in self._loaders.items()}


# Global registry instance
_global_registry = SceneDataLoaderRegistry()


def register_loader(loader: SceneDataLoader, priority: int = 0) -> None:
    """Register a loader with the global registry."""
    _global_registry.register(loader, priority)


def get_loader(name: str) -> Optional[SceneDataLoader]:
    """Get a loader from the global registry."""
    return _global_registry.get_loader(name)


def load_scene(
    source: Union[Path, str, Dict[str, Any]],
    loader_name: Optional[str] = None,
    **kwargs: Any,
) -> SceneData:
    """Load scene data using the global registry."""
    return _global_registry.load(source, loader_name, **kwargs)


def list_loaders() -> List[str]:
    """List all registered loaders."""
    return _global_registry.list_loaders()


# Decorator for auto-registration
def auto_register(priority: int = 0) -> Any:
    """
    Decorator to automatically register a loader class.

    Usage:
        @auto_register(priority=10)
        class MyLoader(SceneDataLoader):
            ...
    """

    def decorator(cls: Type[SceneDataLoader]) -> Type[SceneDataLoader]:
        # Create instance and register
        instance = cls()
        register_loader(instance, priority)
        return cls

    return decorator


# Auto-import all loaders when this module is imported
def _auto_import_loaders() -> None:
    """Automatically import all loader modules to trigger registration."""

    import cosmos_transfer2._src.imaginaire.auxiliary.world_scenario.dataloaders.clipgt_loader
    import cosmos_transfer2._src.imaginaire.auxiliary.world_scenario.dataloaders.rdshq_loader  # noqa: F401


# Automatically import loaders when this module is imported
_auto_import_loaders()
