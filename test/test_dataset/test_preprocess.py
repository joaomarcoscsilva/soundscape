import numpy as np

from soundscape.dataset import preprocess

SR = 22050


def test_convert_to_sample_rate():
    audio = np.random.rand(SR * 5)
    new_audio = preprocess.convert_to_sample_rate(audio, SR, SR * 2)
    assert new_audio.shape == (SR * 5 * 2,)


def test_crop_segment():
    audio = np.random.rand(SR * 5)
    new_audio = preprocess.crop_segment(audio, 1, 2, SR)
    assert new_audio.shape == (SR,)
    assert np.allclose(new_audio, audio[SR : SR * 2])


def test_pad_segment():
    audio = np.random.rand(SR * 2)
    new_audio, start, end = preprocess.pad_segment(audio, 1, 2, SR, 5)
    assert new_audio.shape == (SR * 7,)
    assert start == 3.5
    assert end == 4.5


def test_crop_centered_segment():
    audio = np.random.rand(SR * 2)
    new_audio = preprocess.crop_centered_segment(audio, 1, 2, SR, 5)
    assert new_audio.shape == (SR * 5,)
    assert np.allclose(new_audio[SR * 2 : SR * 3], audio[SR : SR * 2])


def test_generate_melspectrogram():
    spectrogram_kwargs = {
        "n_fft": 2048,
        "hop_length": 256,
        "win_length": 2048,
        "window": "hamming",
        "center": False,
        "pad_mode": None,
        "power": 1.0,
        "n_mels": 256,
        "fmin": 10,
        "fmax": 11025,
        "htk": False,
        "norm": None,
    }

    audio = np.random.rand(SR * 2)
    spectrogram = preprocess.generate_melspectrogram(audio, SR, spectrogram_kwargs)

    assert spectrogram.shape[0] == spectrogram_kwargs["n_mels"]
    assert spectrogram.shape[1] < 1000


def test_process_melspectrogram():
    spectrogram = np.random.rand(256, 1000)
    new_spectrogram = preprocess.process_melspectrogram(spectrogram, 16, [-80, 0])

    assert spectrogram.shape == new_spectrogram.shape
    assert new_spectrogram.dtype == np.uint16
