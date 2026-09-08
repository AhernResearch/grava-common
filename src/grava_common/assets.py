"""Versioned GR-NavSim release manifest; paths are relative to the asset root."""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class SceneAsset:
    id: str
    split: str
    log_name: str
    scene_token: str
    timestamp: str
    image: str
    frame: str
    qa: str | None = None
    annotation: str | None = None
    history: list[str] = field(default_factory=list)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.split not in {"train", "val", "test"}:
            raise ValueError("Expected asset schema_version=1 and split=train/val/test")
        for key in ("id", "log_name", "scene_token", "timestamp", "image", "frame"):
            if not isinstance(getattr(self, key), str) or not getattr(self, key):
                raise ValueError(f"Missing string asset field: {key}")
        for value in [self.image, self.frame, self.qa, self.annotation, *self.history]:
            if value and (Path(value).is_absolute() or ".." in Path(value).parts):
                raise ValueError(f"Asset paths must be relative to their root: {value}")
        if len(self.history) > 3:
            raise ValueError("Expected at most three chronological history frames")

    def path(self, root: Path, name: str) -> Path:
        value = getattr(self, name)
        if not value:
            raise ValueError(f"Asset {self.id} has no {name}")
        path = root / value
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
