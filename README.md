# Whisper data preparation

This project is for my research - fine-tuning whisper model.

The main goal of this project is to provide an interface for loading a file and it's transcriptions and saving them as a dataset of up to 30 seconds segments.

## Usage

As an input you need a list of dicts like this:
```json
[
  {
    "text": "Transcription here",
    "start": 230,
    "end": 540
  },
  {
    "text": "Another segment",
    "start": 1200,
    "end": 3600
  }
]
```
* `text` - transcription
* `start` - start of transcription in miliseconds
* `end` - end of transcription in miliseconds


```python
from whisper_prepare_data import (Processor, save_segments_as_files,
                                  save_as_dataset)

processor = Processor()
data = [
    # list of segments
]
filename = "my_file.mp3"
result = processor(data, filename)
```

Now you can save segmented audio file and its transcription:
```python
save_segments_as_files(result, "segmented_file", "/path/to/location/")
```

or save as dataset:
```python
save_as_dataset(result, "my_dataset", "/path/to/location/")
```
