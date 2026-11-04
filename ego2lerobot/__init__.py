"""Ego dataset conversion utilities.

This package intentionally keeps the heavy dependencies (Ray, h5py, mcap,
LeRobot) optional/lazy so that the execution layer can be reused or imported
without pulling in the full conversion stack.
"""

__version__ = "0.2.0"
