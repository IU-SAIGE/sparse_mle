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

from _models import DenoisingNetwork


#
# parse arguments
#
p = argparse.ArgumentParser()
p.add_argument('-s', '--state_dict_file', type=str, required=True)
p.add_argument('-d', '--device_id', default=F.get_gpu_next_device())
p.add_argument('--disconnect', action='store_true')
args = p.parse_args()
assert os.path.exists(args.state_dict_file)


#
# parse model parameters from filepath
#
output_directory = os.path.dirname(args.state_dict_file)
match = re.search(r'(snr|gen*)\_(?!all)', output_directory)
args.latent_space = 'all'
if match:
    if ('gen' in str(match[1])):
        args.latent_space = 'gender'
    elif ('snr' in str(match[1])):
        args.latent_space = 'snr'

match = re.search(r'(\d+)x(\d)', args.state_dict_file)
args.hidden_size = int(match[1])
args.num_layers = int(match[2])


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

        network = DenoisingNetwork(
            args.hidden_size,
            args.num_layers
        ).to(device=args.device_id)

        network_params = F.count_parameters(network)

        network.load_state_dict(torch.load(
            args.state_dict_file,
            map_location=torch.device(args.device_id),
        ), strict=True)
        network.eval()

        F.write_data(filename=os.path.join(output_directory, 'num_parameters.txt'),
                     data=network_params)

        with torch.cuda.device(args.device_id):
            torch.cuda.empty_cache()

        te_sisdr = dict()

        if args.latent_space in ('gender', 'all'):
            np.random.seed(0)
            torch.manual_seed(0)

            for te_gender in C.gender_all:

                logging.info(f'Now testing model with {te_gender}-gender inputs...')

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

            te_sisdr['mean_gender'] = np.mean([te_sisdr[str(x)] for x in C.gender_all])

        if args.latent_space in ('snr', 'all'):
            np.random.seed(0)
            torch.manual_seed(0)

            for te_snr in C.snr_all:

                logging.info(f'Now testing model with {te_snr} dB mixture SDR inputs...')

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

            te_sisdr['mean_sisdr'] = np.mean([te_sisdr[str(x)] for x in C.snr_all])

        logging.info(json.dumps(te_sisdr, sort_keys=True, indent=4))
        F.write_data(filename=os.path.join(output_directory, f'test_results.txt'),
                     data=te_sisdr)

    return



evaluation()
logging.info('Completed testing Denoising model (with {} hidden units and {} layers).'.format(
    args.hidden_size, args.num_layers))
if args.disconnect:
    time.sleep(60)
    os.kill(os.getppid(), signal.SIGHUP)  # useful for closing tmux sessions

