from dataclasses import dataclass
from math import ceil, floor
from typing import Self, Optional


@dataclass
class AudioSegment:
    text: str
    start: str
    end: str
    start_f: Optional[float] = None
    end_f: Optional[float] = None

    def __post_init__(self):
        self.start_f = float(self.start)
        self.end_f = float(self.end)

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        text = data["text"]
        start = round_partial_floor(float(data["start"]), 0.02)
        end = round_partial_ceil(float(data["end"]), 0.02)

        return AudioSegment(
            text=text, start=f"{start:.2f}", end=f"{end:.2f}"
        )

    def __str__(self):
        return f"<|{self.start}|>{self.text}<|{self.end}|>"

    def str_shifted(self, duration: float) -> str:
        start = self.start_f - duration
        end = self.end_f - duration
        return f"<|{start:.2f}|>{self.text}<|{end:.2f}|>"


class WhisperSegment:
    def __init__(self):
        self.segments: list[AudioSegment] = []
        self.limit = 30.0

    @property
    def segment_start(self) -> Optional[float]:
        if not self.segments:
            return None
        return self.segments[0].start_f

    @property
    def segment_end(self) -> Optional[float]:
        if not self.segments:
            return None
        return self.segments[-1].end_f

    def __str__(self):
        return " ".join([str(s) for s in self.segments])

    def str_shifted(self):
        d = self.segments[0].start_f
        return " ".join([s.str_shifted(d) for s in self.segments])

    def add_segment(self, segment: AudioSegment) -> bool:
        if self.segments and segment.end_f - self.segment_start > self.limit:
            return False
        self.segments.append(segment)
        return True


def round_partial_ceil(value: float, resolution: float) -> float:
    return ceil(round(value, 2) / resolution) * resolution


def round_partial_floor(value: float, resolution: float) -> float:
    return floor(round(value, 2) / resolution) * resolution


def get_audio_segments(data: list[dict]) -> list[AudioSegment]:
    segments = []
    for d in data:
        segments.append(AudioSegment.from_dict(d))
    return segments
