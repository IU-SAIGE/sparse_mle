import datetime
import functools
import json
import os
import subprocess
import re

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torchaudio
from tabulate import tabulate

from argparse import Namespace
from _config import sample_rate as sr
from _config import lookback
from _config import fft_size
from _config import hop_size
from _config import num_samples
from _config import stft_features
from _config import stft_frames
from _config import snr_all
from _config import gender_all
from _config import gender_map


eps = 1e-30


def get_gpu_remaining_memory():
    available_devices = list(map(int, os.environ['CUDA_VISIBLE_DEVICES'].split(',')))
    total_mem = {
        i: round(torch.cuda.get_device_properties(i).total_memory / 1e6)
        for i in range(torch.cuda.device_count())
    }
    result = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.used',
        '--format=csv,nounits,noheader'], encoding='utf-8')
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory = [v for (i, v) in enumerate(gpu_memory) if i in available_devices]
    remaining_mem = {
        i: (v - gpu_memory[i])
        for (i, v) in total_mem.items()
    }
    return remaining_mem


def get_gpu_next_device(buffer_mb=3130):
    device_id = -1
    if torch.cuda.is_available():
        r = get_gpu_remaining_memory()
        i = max(r, key=r.get)
        if r[i] > buffer_mb:
            device_id = i
    return device_id


def get_gender(filepath):
    speaker_id = int(re.match(r'.*?(\d+)\-\d+\-\d+\.flac', filepath)[1])
    return gender_map[speaker_id]


def filter_by_gender(filepaths, gender_val):
    return [f for f in filepaths if get_gender(f) == gender_val]


def initialize_weights(network):
    for name, param in network.named_parameters():
        if 'bias' in name:
            torch.nn.init.constant_(param, 0.0)
        elif 'weight' in name:
            torch.nn.init.xavier_uniform_(param)


def many_to_one(rnn_output, batch_first=True):
    if batch_first:
        o = rnn_output[0][:, -1]
    else:
        o = rnn_output[0][-1]
    return o


def many_to_many(rnn_output):
    o = rnn_output[0]
    return o


def fmt_snr(value):
    if isinstance(value, (np.ndarray, list, tuple)):
        output = "snr_all"
    else:
        prefix = "p" if value >= 0 else "n"
        output = f"snr_{prefix}{abs(value):02}"
    return output


def fmt_gender(value):
    if isinstance(value, (np.ndarray, list, tuple)):
        output = "gender_all"
    else:
        output = f"gender_{value}"
    return output


def fmt_specialty(key):
    return {
        'None' : 'baseline',
        '-5'   : 'snr_n05',
        '0'    : 'snr_p00',
        '5'    : 'snr_p05',
        '10'   : 'snr_p10',
        'M'    : 'gen_M',
        'F'    : 'gen_F',
    }[str(key)]


def fmt_disksize(value, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(value) < 1024.0:
            return ("%3.1f%s%s" % (value, unit, suffix))
        value /= 1024.0
    return ("%.1f%s%s" % (value, 'Yi', suffix))


def count_parameters(network):
    n = sum(p.numel() for p in network.parameters() if p.requires_grad)
    return n


def load_audio(filepath, device, duration=lookback, random_offset=True):
    max_duration = sf.info(filepath).duration
    if random_offset and (max_duration > duration):
        offset = np.random.randint(sr * (max_duration - duration)) // sr
        signal = librosa.core.load(filepath, sr=sr, offset=offset, duration=duration)[0]
    elif (duration == None):
        signal = librosa.core.load(filepath, sr=sr)[0]
    else:
        signal = librosa.core.load(filepath, sr=sr)[0]
        signal = librosa.util.fix_length(signal, size=int(sr * duration))
    signal = torch.from_numpy(signal).to(device)  # convert to torch type
    signal /= (signal.std() + eps)  # standardize to ensure equal loudness
    return signal


def write_image(filename, data):
    plt.subplots(nrows=1, ncols=1, dpi=300)
    plt.imshow(data, aspect='auto', origin='lower')
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight', transparent=True, pad_inches=0)
    plt.close('all')
    return


def write_data(filename, data):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4, sort_keys=True)
    return


def write_table(filename, table_data, headers):
    with open(filename, 'w') as f:
        print(tabulate(table_data, headers=headers), file=f)
    return


def mix_signals(source_signal, noise_signal, snr_db=0):
    mixture_snr = snr_db
    if isinstance(snr_db, (np.ndarray, list, tuple)):
        mixture_snr = np.random.uniform(low=min(snr_db), high=max(snr_db))
    elif isinstance(snr_db, list):
        mixture_snr = np.random.choice(snr_db)
    noise_signal *= (10 ** (-mixture_snr / 20.))
    mixture_signal = source_signal + noise_signal
    return mixture_signal, source_signal, noise_signal


def stft(signal):
    window = torch.hann_window(fft_size, device=signal.device)
    S = torch.stft(signal, n_fft=fft_size, hop_length=hop_size, window=window)
    S_mag = torch.sqrt(S[..., 0] ** 2 + S[..., 1] ** 2)
    return S, S_mag


def istft(spectrogram, mask=None):
    if mask is not None:
        spectrogram[..., 0] *= mask
        spectrogram[..., 1] *= mask
    spectrogram = spectrogram.permute(0, 2, 1, 3)
    window = torch.hann_window(fft_size, device=spectrogram.device)
    y = torchaudio.functional.istft(spectrogram, n_fft=fft_size, hop_length=hop_size, window=window)
    return y


def logistic(x, L=1, k=1, x_o=0):
    return L / (1 + np.exp(-k * (x - x_o)))


def calculate_accuracy(source_onehot, estimated_softmax):
    x = source_onehot
    y = estimated_softmax

    # add a batch axis if non-existant
    if len(x.shape) != 2:
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)

    count = torch.sum(
        x.argmax(dim=1) == y.argmax(dim=1)
    )
    total = x.shape[0]
    accuracy = float(count) / float(total)

    return accuracy


def calculate_sdr(source_signal, estimated_signal, offset=None, scale_invariant=False):
    s = source_signal
    y = estimated_signal

    # add a batch axis if non-existant
    if len(s.shape) != 2:
        s = s.unsqueeze(0)
        y = y.unsqueeze(0)

    # truncate all signals in the batch to match the minimum-length
    min_length = min(s.shape[-1], y.shape[-1])
    s = s[..., :min_length]
    y = y[..., :min_length]

    if scale_invariant:
        alpha = s.mm(y.T).diag()
        alpha /= ((s ** 2).sum(dim=1) + eps)
        alpha = alpha.unsqueeze(1)  # to allow broadcasting
    else:
        alpha = 1

    e_target = s * alpha
    e_res = e_target - y

    numerator = (e_target ** 2).sum(dim=1)
    denominator = (e_res ** 2).sum(dim=1) + eps
    sdr = 10 * torch.log10((numerator / denominator) + eps)

    # if `offset` is non-zero, this function returns the relative SDR
    # improvement for each signal in the batch
    if offset is not None:
        sdr -= offset

    return sdr


def calculate_sisdr(source_signal, estimated_signal, offset=None):
    return calculate_sdr(source_signal, estimated_signal, offset, True)


def loss_sdr(source_signal, estimated_signal):
    return -1.*torch.mean(calculate_sdr(source_signal, estimated_signal))


def loss_sisdr(source_signal, estimated_signal):
    return -1.*torch.mean(calculate_sisdr(source_signal, estimated_signal))


def calculate_mse(source, estimate):
    return torch.nn.functional.mse_loss(estimate, source, reduction='mean')


def calculate_bce(source, estimate):
    return torch.nn.functional.binary_cross_entropy(estimate, source, reduction='mean')


def calculate_masking_target(source_magnitude, noise_magnitude, beta=2):
    s = (source_magnitude ** beta)
    n = (noise_magnitude ** beta)
    y = torch.sqrt(s / (s + n + eps))
    return y


def generate_filepaths(
    tr_speech_dir='/media/sdc1/librispeech/train-clean-100/',
    te_speech_dir='/media/sdc1/librispeech/test-clean/',
    tr_noise_dir='/media/sdc1/musan/noise/free-sound/',
    te_noise_dir='/media/sdc1/musan/noise/sound-bible/',
    tr_num_speakers=None,
    te_num_speakers=None,
    tr_num_noises=None,
    te_num_noises=None,
    vl_fold_percentage=0.05,
    out_file='filepaths.npy'
):
    tr_speakers = [d for d in os.scandir(tr_speech_dir) if d.is_dir()]
    tr_speakers = tr_speakers[:tr_num_speakers]
    te_speakers = [d for d in os.scandir(te_speech_dir) if d.is_dir()]
    te_speakers = te_speakers[:te_num_speakers]
    tr_utterances = sum([librosa.util.find_files(d) for d in tr_speakers], [])
    tr_noises = librosa.util.find_files(tr_noise_dir)[:tr_num_noises]
    te_utterances = sum([librosa.util.find_files(d) for d in te_speakers], [])
    te_noises = librosa.util.find_files(te_noise_dir)[:te_num_noises]

    np.random.seed(0)
    tr_utterances = np.random.permutation(tr_utterances)
    tr_noises = np.random.permutation(tr_noises)

    vl_utterances = list()
    vl_noises = list()
    if vl_fold_percentage > 0:
        l_u = round(vl_fold_percentage*len(tr_utterances))
        l_n = round(vl_fold_percentage*len(tr_noises))
        (tr_utterances, vl_utterances) = np.split(tr_utterances, [-l_u])
        (tr_noises, vl_noises) = np.split(tr_noises, [-l_n])

    np.random.seed(0)
    te_utterances = np.random.permutation(te_utterances)
    te_noises = np.random.permutation(te_noises)

    np.save(out_file, (tr_utterances, tr_noises,
        vl_utterances, vl_noises,
        te_utterances, te_noises))
    return


def generate_batch(files_speech, files_noise, device, mixture_snr=snr_all):

    assert len(files_speech) == len(files_noise)
    batch_size = len(files_speech)

    # instantiate waveforms
    speech_waveform = torch.zeros(batch_size, num_samples, device=device)
    noise_waveform = torch.zeros(batch_size, num_samples, device=device)
    mixture_waveform = torch.zeros(batch_size, num_samples, device=device)

    # instantiate spectrograms
    speech_magnitude = torch.zeros(batch_size, stft_frames, stft_features, device=device)
    noise_magnitude = torch.zeros(batch_size, stft_frames, stft_features, device=device)
    mixture_magnitude = torch.zeros(batch_size, stft_frames, stft_features, device=device)
    mixture_stft = torch.zeros(batch_size, stft_frames, stft_features, 2, device=device)

    # instantiate targets
    mask = torch.zeros(batch_size, stft_frames, stft_features, device=device)
    actual_sdr = torch.zeros(batch_size, device=device)
    actual_sisdr = torch.zeros(batch_size, device=device)
    specialist_index_gender = torch.zeros(batch_size, len(gender_all), device=device)
    specialist_index_sdr = torch.zeros(batch_size, len(snr_all), device=device)

    for (i, fs, fn) in zip(range(batch_size), files_speech, files_noise):

        # mix the signals up
        source = load_audio(fs, device=device)
        noise = load_audio(fn, device=device)
        (x, s, n) = mix_signals(source, noise, snr_db=mixture_snr)
        (S, S_mag) = stft(s)
        (N, N_mag) = stft(n)
        (X, X_mag) = stft(x)
        (M) = calculate_masking_target(S_mag, N_mag)

        # store the variables
        speech_waveform[i] = s
        noise_waveform[i] = n
        mixture_waveform[i] = x
        mixture_stft[i] = X.permute(1, 0, 2)[:stft_frames] # (seq_len, num_features, channel)
        speech_magnitude[i] = S_mag.permute(1, 0)[:stft_frames]  # (seq_len, num_features)
        noise_magnitude[i] = N_mag.permute(1, 0)[:stft_frames]  # (seq_len, num_features)
        mixture_magnitude[i] = X_mag.permute(1, 0)[:stft_frames]  # (seq_len, num_features)
        mask[i] = M.permute(1, 0)[:stft_frames]  # (seq_len, num_features)
        actual_sdr[i] = float(calculate_sdr(s, x).item())
        actual_sisdr[i] = float(calculate_sisdr(s, x).item())
        gender_index = int(get_gender(fs)=='F')
        sdr_index = int(np.abs(snr_all - actual_sdr[i].item()).argmin())
        specialist_index_gender[i][gender_index] = 1
        specialist_index_sdr[i][sdr_index] = 1

    return Namespace(
        s=speech_waveform,
        n=noise_waveform,
        x=mixture_waveform,
        X=mixture_stft,
        S_mag=speech_magnitude,
        N_mag=noise_magnitude,
        X_mag=mixture_magnitude,
        M=mask,
        actual_sdr=actual_sdr,
        actual_sisdr=actual_sisdr,
        index_gender=specialist_index_gender,
        index_sdr=specialist_index_sdr
    )
