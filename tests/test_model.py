from whisper_prepare_data import model


class TestAudioSegment:
    def test_model_audio_segment_single_word(self):
        data = {"text": "test", "start": 234, "end": 345}
        segment = model.AudioSegment.from_dict(data)

        expected = model.AudioSegment(text="test", start=220, end=340)

        assert segment == expected
        assert str(segment) == "<|0.22|>test<|0.34|>"

    def test_model_get_sudio_segments_single_segment(self):
        data = [{"text": "test", "start": 234, "end": 345}]

        segments = model.get_audio_segments(data)

        expected = [model.AudioSegment(text="test", start=220, end=340)]

        assert segments == expected

    def test_model_get_audio_segments_multiple_segments(self):
        data = [
            {"text": "test", "start": 234, "end": 345},
            {"text": "test 2", "start": 15320, "end": 17350}
        ]

        segments = model.get_audio_segments(data)

        expected = [
            model.AudioSegment(text="test", start=220, end=340),
            model.AudioSegment(text="test 2", start=15320, end=17360)
        ]

        assert segments == expected

    def test_model_audio_segment_shifted(self):
        segment = model.AudioSegment(text="test 2", start=15320, end=17360)

        shifted_str = segment.str_shifted(10300)

        assert shifted_str == "<|5.02|>test 2<|7.06|>"


class TestWhisperSegment:
    def test_add_single_segment(self):
        data = {"text": "test", "start": 234, "end": 345}
        segment = model.AudioSegment.from_dict(data)

        ws = model.WhisperSegment()
        added = ws.add_segment(segment)

        assert added
        assert len(ws.segments) == 1

    def test_add_multiple_segments_within_limit(self):
        data = [
            {"text": "test", "start": 234, "end": 345},
            {"text": "test 2", "start": 15320, "end": 17350}
        ]
        segments = model.get_audio_segments(data)
        ws = model.WhisperSegment()

        for s in segments:
            assert ws.add_segment(s)
        assert len(ws.segments) == 2

    def test_add_segments_out_of_limit(self):
        data = [
            {"text": "test 2", "start": 320, "end": 17350},
            {"text": "test 30sec", "start": 38320, "end": 40000},
        ]
        segments = model.get_audio_segments(data)
        ws = model.WhisperSegment()

        assert ws.add_segment(segments[0])
        assert not ws.add_segment(segments[1])
        assert len(ws.segments) == 1

    def test_shifted_segments(self):
        segments = [
            model.AudioSegment(text="test", start=9220, end=12340),
            model.AudioSegment(text="test 2", start=15320, end=17360)
        ]

        ws = model.WhisperSegment()
        for s in segments:
            ws.add_segment(s)

        shifted_str = ws.str_shifted()

        assert shifted_str == "<|0.00|>test<|3.12|> <|6.10|>test 2<|8.14|>"
