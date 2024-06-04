from dataclasses import dataclass
from math import ceil, floor
from typing import Self, Optional


@dataclass
class AudioSegment:
    text: str
    start: int
    end: int
    start_str: Optional[str] = None
    end_str: Optional[str] = None

    def __post_init__(self):
        self.start_str = f"{(self.start / 1000.0):.2f}"
        self.end_str = f"{(self.end / 1000.0):.2f}"

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        text = data["text"]
        start_i = int(data["start"])
        end_i = int(data["end"])
        start_i -= start_i % 10
        end_i -= end_i % 10
        start = round_partial_floor_int(start_i, 20)
        end = round_partial_ceil_int(end_i, 20)

        return AudioSegment(text=text, start=start, end=end)

    def __str__(self):
        return f"<|{self.start_str}|>{self.text}<|{self.end_str}|>"

    def str_shifted(self, duration: int) -> str:
        start = self.start - duration
        end = self.end - duration
        return f"<|{(start/1000.0):.2f}|>{self.text}<|{(end/1000.0):.2f}|>"


class WhisperSegment:
    def __init__(self):
        self.segments: list[AudioSegment] = []
        self.limit = 30000

    @property
    def segment_start(self) -> Optional[float]:
        if not self.segments:
            return None
        return self.segments[0].start

    @property
    def segment_end(self) -> Optional[float]:
        if not self.segments:
            return None
        return self.segments[-1].end

    def __str__(self):
        return "".join([str(s) for s in self.segments])

    def str_shifted(self):
        d = self.segments[0].start
        return "".join([s.str_shifted(d) for s in self.segments])

    def add_segment(self, segment: AudioSegment) -> bool:
        if self.segments and segment.end - self.segment_start > self.limit:
            return False
        self.segments.append(segment)
        return True


def round_partial_ceil(value: float, resolution: float) -> float:
    return ceil(round(value, 2) / resolution) * resolution


def round_partial_floor(value: float, resolution: float) -> float:
    return floor(round(value, 2) / resolution) * resolution


def round_partial_ceil_int(value: int, resolution: int) -> int:
    return int(ceil(value / resolution)) * resolution


def round_partial_floor_int(value: int, resolution: int) -> int:
    return int(floor(value / resolution)) * resolution


def get_audio_segments(data: list[dict]) -> list[AudioSegment]:
    segments = []
    for d in data:
        segments.append(AudioSegment.from_dict(d))
    return segments
