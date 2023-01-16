import numpy as np
from numpy.fft import fft


def get_mel_filters(sr, n_fft, n_mels=128, dtype=np.float32):
    # Initialize the weights
    n_mels = int(n_mels)
    weights = np.zeros((n_mels, int(1 + n_fft // 2)), dtype=dtype)

    # Center freqs of each FFT bin
    fftfreqs = np.fft.rfftfreq(n=n_fft, d=1.0 / sr)

    # 'Center freqs' of mel bands - uniformly spaced between limits
    min_mel = 0.0
    max_mel = 45.245640471924965

    mels = np.linspace(min_mel, max_mel, n_mels + 2)

    mels = np.asanyarray(mels)

    # Fill in the linear scale
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels

    # And now the nonlinear scale
    min_log_hz = 1000.0  # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
    logstep = np.log(6.4) / 27.0  # step size for log region

    # If we have vector data, vectorize
    log_t = mels >= min_log_mel
    freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))

    mel_f = freqs

    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)

    for i in range(n_mels):
        # lower and upper slopes for all bins
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]

        # .. then intersect them with each other and zero
        weights[i] = np.maximum(0, np.minimum(lower, upper))

    # Slaney-style mel is scaled to be approx constant energy per channel
    enorm = 2.0 / (mel_f[2: n_mels + 2] - mel_f[:n_mels])
    weights *= enorm[:, np.newaxis]

    return weights


def stft_fun(frames, window, n_fft):
    """
    Calculates the complex Short-Time Fourier Transform (STFT) of the given framed signal. Should give the same
    results as `torch.stft`.
    """
    frame_size = frames.shape[1]
    fft_size = n_fft

    if fft_size is None:
        fft_size = frame_size

    if fft_size < frame_size:
        raise ValueError("FFT size must greater or equal the frame size")
    # number of FFT bins to store
    num_fft_bins = (fft_size >> 1) + 1

    data = np.empty((len(frames), num_fft_bins), dtype=np.complex64)
    fft_signal = np.zeros(fft_size)

    for f, frame in enumerate(frames):
        if window is not None:
            np.multiply(frame, window, out=fft_signal[:frame_size])
        else:
            fft_signal[:frame_size] = frame
        data[f] = fft(fft_signal, axis=0)[:num_fft_bins]
    return data.T


def fram_wave(waveform, n_fft, hop_length, center=True):
    """
    Transform a raw waveform into a list of smaller waveforms. The window length defines how much of the signal is
    contain in each frame (smalle waveform), while the hope length defines the step between the beginning of each
    new frame.

    Centering is done by reflecting the waveform which is first centered around `frame_idx * hop_length`.
    """
    frames = []
    for i in range(0, waveform.shape[0] + 1, hop_length):

        half_window = (n_fft - 1) // 2 + 1
        if center:
            start = i - half_window if i > half_window else 0
            end = i + half_window if i < waveform.shape[0] - half_window else waveform.shape[0]

            frame = waveform[start:end]

            if start == 0:
                padd_width = (-i + half_window, 0)
                frame = np.pad(frame, pad_width=padd_width, mode="reflect")

            elif end == waveform.shape[0]:
                padd_width = (0, (i - waveform.shape[0] + half_window))
                frame = np.pad(frame, pad_width=padd_width, mode="reflect")

        else:
            frame = waveform[i: i + n_fft]
            frame_width = frame.shape[0]
            if frame_width < waveform.shape[0]:
                frame = np.lib.pad(
                    frame, pad_width=(0, n_fft - frame_width), mode="constant", constant_values=0
                )

        frames.append(frame)
    return np.stack(frames, 0)


def _np_extract_fbank_features(waveform: np.array, n_fft, sr, n_mels, hop_length, center) -> np.ndarray:
    """
    Compute the log-Mel spectrogram of the provided audio, gives similar results whisper's original torch
    implementation with 1e-5 tolerance.
    """
    window = np.hanning(n_fft + 1)[:-1]

    frames = fram_wave(waveform, n_fft, hop_length, center)
    stft = stft_fun(frames, window=window, n_fft=n_fft)
    magnitudes = np.abs(stft[:, :-1]) ** 2

    filters = get_mel_filters(sr, n_fft, n_mels)
    mel_spec = filters @ magnitudes

    log_spec = np.log10(np.clip(mel_spec, a_min=1e-10, a_max=None))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0

    return log_spec


def whisper_audio2mel(raw_speech, sampling_rate, n_fft, hop_length, center, n_mels):

    # raw_speech = [np.asarray([speech], dtype=np.float32).T for speech in raw_speech]
    # raw_speech = np.expand_dims(raw_speech, axis=0)
    # waveform = raw_speech[0]
    #
    # input_features = [_np_extract_fbank_features(waveform, n_fft, sampling_rate, n_mels, hop_length, center)]
    input_features = _np_extract_fbank_features(raw_speech, n_fft, sampling_rate, n_mels, hop_length, center)

    return input_features