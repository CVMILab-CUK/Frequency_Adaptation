"""Trainers are imported lazily: `sdxl_trainer` / `ddp_trainer` pull in optional
heavy dependencies, and importing the package should not require them."""
from .base_trainer import BaseTrainer

__all__ = ["BaseTrainer"]
