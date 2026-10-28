"""Environment boundaries. Importing this package does not import MuJoCo."""

from disco_spatial.envs.memory_maze import RGBOnlyMemoryMaze, make_memory_maze

__all__ = ["RGBOnlyMemoryMaze", "make_memory_maze"]
