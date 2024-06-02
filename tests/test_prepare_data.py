import json
import os
import pytest
import librosa
import numpy as np
from datasets import load_from_disk

from whisper_prepare_data import (Processor, save_segments_as_files,
                                  save_as_dataset)

dir_path = os.path.dirname(os.path.realpath(__file__))


@pytest.fixture
def sample_data() -> list[dict]:
    with open(f"{dir_path}/resources/hr.json") as f:
        data = json.loads(f.read())

    return [
        {
            "text": segment["text"],
            "start": int(segment["start"] * 1000),
            "end": int(segment["end"] * 1000)
        }
        for segment in data["segments"]
    ]


class TestProcessor:
    def test_processor_sample_audio(self, sample_data):
        filename = f"{dir_path}/resources/hr.mp3"
        processor = Processor()

        result = processor(sample_data, filename)

        assert result

        for r in result:
            assert len(r["audio"]) <= 480000

        assert "Ja p... dole!" in result[-1]["text"]


def test_save_segments_as_files(sample_data, tmp_path):
    filename = f"{dir_path}/resources/hr.mp3"
    processor = Processor()

    result = processor(sample_data, filename)

    save_segments_as_files(result, "test_segment", str(tmp_path))

    assert len(os.listdir(tmp_path/"test_segment")) == 20

    for i, segment in enumerate(result):
        segment_path = f"{tmp_path}/test_segment/segment_{i:03d}"
        audio, _ = librosa.load(
            f"{segment_path}.wav",
            sr=16000,
            dtype=np.float32
        )
        with open(f"{segment_path}.txt", encoding="utf-8") as f:
            text = f.read()

        assert len(audio) == len(segment["audio"])
        assert text == segment["text"]


def test_save_as_dataset(sample_data, tmp_path):
    dataset_name = "test_dataset"
    filename = f"{dir_path}/resources/hr.mp3"
    processor = Processor()
    result = processor(sample_data, filename)

    save_as_dataset(result, dataset_name, str(tmp_path))

    dataset = load_from_disk(str(tmp_path/dataset_name))

    assert dataset
    assert len(dataset) == 10
