import json
import os
import pytest

from whisper_prepare_data import Processor

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
