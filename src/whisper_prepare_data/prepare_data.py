import librosa
import numpy as np

from whisper_prepare_data.model import (get_audio_segments, WhisperSegment)


class Processor:
    def __call__(self, data: list[dict], filename) -> list[dict]:
        result: list[dict] = []
        segments = get_audio_segments(data)
        samplerate = 16000
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
                    r = segments[i].start_f - e
                    max_d = tmp_ws.limit - d
                    e = e + max(0.0, min(max_d, r - 0.02))
                    assert e <= segments[i].start_f
                arr = audio[int(s * samplerate):int(e * samplerate)]
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
                e = min(e + max_d, len(audio)/samplerate)
                arr = audio[int(s * samplerate):int(e * samplerate)]
                result.append(
                    {"text": tmp_ws.str_shifted(), "audio": arr}
                )
            i += 1

        return result
