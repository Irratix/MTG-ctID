from dataclasses import dataclass, field
from typing import Any

@dataclass
class ModelConfig:
    # A model and all its training settings
    name: str
    model: Any
    optimizer: Any
    scheduler: Any
    checkpoint_folder: str = "models"
    num_epochs: int = 50
    unfreeze_schedule: dict = field(default_factory=dict)

