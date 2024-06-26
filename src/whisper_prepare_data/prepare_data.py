import os

import librosa
import soundfile as sf
import numpy as np
from datasets import Dataset
import pandas as pd

from whisper_prepare_data.model import (get_audio_segments, WhisperSegment)


class Processor:
    def __call__(self, data: list[dict], filename) -> list[dict]:
        result: list[dict] = []
        segments = get_audio_segments(data)
        samplerate = 16000
        sr_sec = samplerate / 1000
        audio, _ = librosa.load(filename, sr=samplerate, dtype=np.float32)

        i = 0
        tmp_ws = WhisperSegment()
        while i < len(segments):
            is_added = tmp_ws.add_segment(segments[i])
            if not is_added:
                s = tmp_ws.segment_start
                e = tmp_ws.segment_end
                d = e - s
                if d < tmp_ws.limit:
                    r = segments[i].start - e
                    max_d = tmp_ws.limit - d
                    e = e + max(0.0, min(max_d, r - 20))
                    assert e <= segments[i].start
                arr = audio[int(s * sr_sec):int(e * sr_sec)]
                result.append(
                    {"text": tmp_ws.str_shifted(), "audio": arr}
                )
                tmp_ws = WhisperSegment()
                continue
            elif i == len(segments) - 1:
                s = tmp_ws.segment_start
                e = tmp_ws.segment_end
                d = e - s
                max_d = tmp_ws.limit - d
                e = min(e + max_d, len(audio)/sr_sec)
                arr = audio[int(s * sr_sec):int(e * sr_sec)]
                result.append(
                    {"text": tmp_ws.str_shifted(), "audio": arr}
                )
            i += 1

        return result


def save_segments_as_files(
        segments: list[dict], segment_name: str, save_dir: str
) -> None:
    os.makedirs(f"{save_dir}/{segment_name}", exist_ok=True)
    for i, segment in enumerate(segments):
        sf.write(
            file=f"{save_dir}/{segment_name}/segment_{i:03d}.wav",
            data=segment["audio"],
            samplerate=16000,
            subtype="PCM_24"
        )
        with open(
                f"{save_dir}/{segment_name}/segment_{i:03d}.txt", "w",
                encoding="utf-8"
        ) as f:
            f.write(segment["text"])


def save_as_dataset(
        segment_list: list[dict], dataset_name: str, save_dir: str
) -> None:
    os.makedirs(f"{save_dir}", exist_ok=True)
    dataset = Dataset.from_pandas(pd.DataFrame(data=segment_list))
    dataset.save_to_disk(f"{save_dir}/{dataset_name}")
