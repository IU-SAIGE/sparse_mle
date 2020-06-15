import argparse
import os
import time
import logging
import signal
import json
import re

import numpy as np
import torch

from math import ceil

import _config as C
import _functions as F

from _models import EnsembleNetwork


#
# parse arguments
#
p = argparse.ArgumentParser()
p.add_argument('-sg', '--state_dict_file_gating', type=str, required=True)
p.add_argument('-ss', '--state_dict_file_specialist', type=str, required=True)
p.add_argument('-d', '--device_id', default=F.get_gpu_next_device())
p.add_argument('--disconnect', action='store_true')
args = p.parse_args()

assert os.path.exists(args.state_dict_file_gating)
assert os.path.exists(args.state_dict_file_specialist)


#
# parse model parameters from filepath
#
match = re.search(r'\/(\d+)x(\d+)\/', args.state_dict_file_gating)
hidden_size_gating = int(match[1])
num_layers_gating = int(match[2])
architecture_gating = f'{match[1]}x{match[2]}'

match = re.search(r'\/(\d+)x(\d+)\/', args.state_dict_file_specialist)
hidden_size_specialist = int(match[1])
num_layers_specialist = int(match[2])
architecture_specialist = f'{match[1]}x{match[2]}'

match = re.search(r'(snr|gen*)\_(?!all)', args.state_dict_file_specialist)
args.latent_space = 'snr'
if match:
    if ('gen' in str(match[1])):
        args.latent_space = 'gender'

output_directory = C.results_directory + '/Ensemble_without_FT/' \
                   f'/{args.latent_space}' \
                   f'/g{hidden_size_gating:04}x{num_layers_gating}/' \
                   f'/s{hidden_size_specialist:04}x{num_layers_specialist}/'
output_directory = os.path.abspath(os.path.expanduser(output_directory))
os.makedirs(output_directory, exist_ok=True)

file_gating = args.state_dict_file_gating
if args.latent_space == 'snr':
    files_specialists = [
        re.sub(r'snr_[np]\d+', F.fmt_snr(i), args.state_dict_file_specialist)
        for i in C.snr_all
    ]
elif args.latent_space == 'gender':
    files_specialists = [
        re.sub(r'gen_[MF]', 'gen_M', args.state_dict_file_specialist),
        re.sub(r'gen_[MF]', 'gen_F', args.state_dict_file_specialist),
    ]


#
# define logger
#
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [PID %(process)d] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(os.path.join(output_directory, 'testing.log')),
        logging.StreamHandler(),
    ],
)


#
# load audio filepaths
#
filepaths = np.load('filepaths.npy', allow_pickle=True)
(te_utterances, te_noises) = filepaths[4:6]


#
# evaluation
#

def evaluation():
    with torch.no_grad():

        #
        # initialize network
        #
        np.random.seed(0)
        torch.manual_seed(0)

        network_params = 0

        network = EnsembleNetwork(
            filepath_gating=file_gating,
            filepaths_denoising=files_specialists,
            g_hs=hidden_size_gating,
            g_nl=num_layers_gating,
            s_hs=hidden_size_specialist,
            s_nl=num_layers_specialist,
            ct=args.latent_space,
        ).to(device=args.device_id)

        F.write_data(filename=os.path.join(output_directory, 'files_gating.txt'),
                     data=str(file_gating))
        F.write_data(filename=os.path.join(output_directory, 'files_specialist.txt'),
                     data=str(files_specialists))

        with torch.cuda.device(args.device_id):
            torch.cuda.empty_cache()


        if args.latent_space == 'gender':

            te_sisdr = {str(k): 0 for k in C.gender_all}
            for te_gender in C.gender_all:

                te_batch_durations = list()
                te_batch_sisdr = list()
                files_speech = np.random.choice(F.filter_by_gender(
                    te_utterances,
                    te_gender
                ), size=C.te_batch_size)
                files_noise = np.random.choice(te_noises, size=C.te_batch_size)

                for (i, fs, fn) in zip(range(C.te_batch_size), files_speech, files_noise):

                    source = F.load_audio(fs, duration=None, random_offset=False, device=args.device_id)
                    noise = F.load_audio(fn, duration=None, random_offset=False, device=args.device_id)
                    min_length = min(len(source), len(noise))
                    stft_frames = ceil(min_length/C.hop_size)
                    source = source[:min_length]
                    noise = noise[:min_length]

                    (x, s, n) = F.mix_signals(source, noise, snr_db=C.snr_all)
                    (X, X_mag) = F.stft(x)

                    X = X.permute(1, 0, 2)[:stft_frames] # (seq_len, num_features, channel)
                    X_mag = X_mag.permute(1, 0)[:stft_frames]  # (seq_len, num_features)
                    X = torch.unsqueeze(X, dim=0)
                    X_mag = torch.unsqueeze(X_mag, dim=0)
                    s = torch.unsqueeze(s, dim=0)
                    x = torch.unsqueeze(x, dim=0)

                    actual_sisdr = float(F.calculate_sisdr(s, x).item())

                    # feed-forward
                    M_hat = network(X_mag)
                    s_hat = F.istft(X, mask=M_hat)

                    te_batch_sisdr.append((F.calculate_sisdr(s, s_hat, offset=actual_sisdr).mean().item()))
                    te_batch_durations.append(min_length)

                # store the weighted average results
                te_sisdr[str(te_gender)] = np.average(te_batch_sisdr, weights=te_batch_durations)

        elif args.latent_space == 'snr':

            te_sisdr = {str(k): 0 for k in C.snr_all}
            for te_snr in C.snr_all:

                te_batch_durations = list()
                te_batch_sisdr = list()
                files_speech = np.random.choice(te_utterances, size=C.te_batch_size)
                files_noise = np.random.choice(te_noises, size=C.te_batch_size)

                for (i, fs, fn) in zip(range(C.te_batch_size), files_speech, files_noise):

                    source = F.load_audio(fs, duration=None, random_offset=False, device=args.device_id)
                    noise = F.load_audio(fn, duration=None, random_offset=False, device=args.device_id)
                    min_length = min(len(source), len(noise))
                    stft_frames = ceil(min_length/C.hop_size)
                    source = source[:min_length]
                    noise = noise[:min_length]

                    (x, s, n) = F.mix_signals(source, noise, snr_db=te_snr)
                    (X, X_mag) = F.stft(x)

                    X = X.permute(1, 0, 2)[:stft_frames] # (seq_len, num_features, channel)
                    X_mag = X_mag.permute(1, 0)[:stft_frames]  # (seq_len, num_features)
                    X = torch.unsqueeze(X, dim=0)
                    X_mag = torch.unsqueeze(X_mag, dim=0)
                    s = torch.unsqueeze(s, dim=0)
                    x = torch.unsqueeze(x, dim=0)

                    actual_sisdr = float(F.calculate_sisdr(s, x).item())

                    # feed-forward
                    M_hat = network(X_mag)
                    s_hat = F.istft(X, mask=M_hat)

                    te_batch_sisdr.append((F.calculate_sisdr(s, s_hat, offset=actual_sisdr).mean().item()))
                    te_batch_durations.append(min_length)

                # store the weighted average results
                te_sisdr[str(te_snr)] = np.average(te_batch_sisdr, weights=te_batch_durations)

        te_sisdr['mean'] = np.mean(list(te_sisdr.values()))

        logging.info(json.dumps(te_sisdr, sort_keys=True, indent=4))
        F.write_data(filename=os.path.join(output_directory, f'test_results.txt'),
                     data=te_sisdr)

    return



evaluation()
logging.info('Completed testing Ensemble model without fine-tuning (with Gating architecture {} and Specialist architecture {}).'.format(
    architecture_gating, architecture_specialist))
if args.disconnect:
    time.sleep(60)
    os.kill(os.getppid(), signal.SIGHUP)  # useful for closing tmux sessions

