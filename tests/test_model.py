from whisper_prepare_data import model


class TestAudioSegment:
    def test_model_audio_segment_single_word(self):
        data = {"text": "test", "start": "0.234", "end": "0.345"}
        segment = model.AudioSegment.from_dict(data)

        expected = model.AudioSegment(
            text="test", start="0.22", end="0.34"
        )

        assert segment == expected
        assert str(segment) == "<|0.22|>test<|0.34|>"

    def test_model_get_sudio_segments_single_segment(self):
        data = [{"text": "test", "start": "0.234", "end": "0.345"}]

        segments = model.get_audio_segments(data)

        expected = [model.AudioSegment(
            text="test", start="0.22", end="0.34"
        )]

        assert segments == expected

    def test_model_get_audio_segments_multiple_segments(self):
        data = [
            {"text": "test", "start": "0.234", "end": "0.345"},
            {"text": "test 2", "start": "15.32", "end": "17.35"}
        ]

        segments = model.get_audio_segments(data)

        expected = [
            model.AudioSegment(text="test", start="0.22", end="0.34"),
            model.AudioSegment(text="test 2", start="15.32", end="17.36")
        ]

        assert segments == expected

    def test_model_audio_segment_shifted(self):
        segment = model.AudioSegment(text="test 2", start="15.32", end="17.36")

        shifted_str = segment.str_shifted(10.3)

        assert shifted_str == "<|5.02|>test 2<|7.06|>"


class TestWhisperSegment:
    def test_add_single_segment(self):
        data = {"text": "test", "start": "0.234", "end": "0.345"}
        segment = model.AudioSegment.from_dict(data)

        ws = model.WhisperSegment()
        added = ws.add_segment(segment)

        assert added
        assert len(ws.segments) == 1

    def test_add_multiple_segments_within_limit(self):
        data = [
            {"text": "test", "start": "0.234", "end": "0.345"},
            {"text": "test 2", "start": "15.32", "end": "17.35"}
        ]
        segments = model.get_audio_segments(data)
        ws = model.WhisperSegment()

        for s in segments:
            assert ws.add_segment(s)
        assert len(ws.segments) == 2

    def test_add_segments_out_of_limit(self):
        data = [
            {"text": "test 2", "start": "0.32", "end": "17.35"},
            {"text": "test 30sec", "start": "38.32", "end": "40.00"},
        ]
        segments = model.get_audio_segments(data)
        ws = model.WhisperSegment()

        assert ws.add_segment(segments[0])
        assert not ws.add_segment(segments[1])
        assert len(ws.segments) == 1

    def test_shifted_segments(self):
        segments = [
            model.AudioSegment(text="test", start="9.22", end="12.34"),
            model.AudioSegment(text="test 2", start="15.32", end="17.36")
        ]

        ws = model.WhisperSegment()
        for s in segments:
            ws.add_segment(s)

        shifted_str = ws.str_shifted()

        assert shifted_str == "<|0.00|>test<|3.12|> <|6.10|>test 2<|8.14|>"
