# dubber/models.py
from dataclasses import dataclass, asdict


@dataclass
class Segment:
    start: float
    end: float
    speaker_id: str
    gender: str = "unknown"
    text: str = ""
    text_tr: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Segment":
        return cls(**d)
