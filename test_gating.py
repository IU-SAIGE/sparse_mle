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

from _models import GatingNetwork


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
match = re.search(r'(snr|gender)', output_directory)
args.latent_space = str(match[1])
args.num_clusters = C.num_clusters[args.latent_space]

match = re.search(r'(\d+)x(\d+)', output_directory)
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

        network = GatingNetwork(
            args.hidden_size,
            args.num_layers,
            args.num_clusters,
        ).to(device=args.device_id)

        network_params = F.count_parameters(network)

        network.load_state_dict(torch.load(
            args.state_dict_file,
            map_location=torch.device(args.device_id),
        ), strict=False)
        network.eval()

        F.write_data(filename=os.path.join(output_directory, 'num_parameters.txt'),
                     data=network_params)

        with torch.cuda.device(args.device_id):
            torch.cuda.empty_cache()


        if args.latent_space == 'gender':

            te_accuracy = {str(k): 0 for k in C.gender_all}
            for te_gender in C.gender_all:

                te_batch_durations = list()
                te_batch_accuracy = list()
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

                    X_mag = X_mag.permute(1, 0)[:stft_frames]  # (seq_len, num_features)
                    X_mag = torch.unsqueeze(X_mag, dim=0)

                    Y = torch.zeros(1, len(C.gender_all), device=args.device_id)
                    gender_index = int(te_gender == 'F')
                    Y[..., gender_index] = 1

                    # forward pass
                    Y_hat = network(X_mag)

                    te_batch_accuracy.append(F.calculate_accuracy(Y, Y_hat))
                    te_batch_durations.append(min_length)

                # store the weighted average results
                te_accuracy[str(te_gender)] = np.average(te_batch_accuracy, weights=te_batch_durations)

        elif args.latent_space == 'snr':

            te_accuracy = {str(k): 0 for k in C.snr_all}
            for te_snr in C.snr_all:

                te_batch_durations = list()
                te_batch_accuracy = list()
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

                    X_mag = X_mag.permute(1, 0)[:stft_frames]  # (seq_len, num_features)
                    X_mag = torch.unsqueeze(X_mag, dim=0)

                    actual_sdr = float(F.calculate_sdr(s, x).item())
                    sdr_index = int(np.abs(C.snr_all - actual_sdr).argmin())
                    Y = torch.zeros(1, len(C.snr_all), device=args.device_id)
                    Y[..., sdr_index] = 1

                    # forward pass
                    Y_hat = network(X_mag)

                    te_batch_accuracy.append(F.calculate_accuracy(Y, Y_hat))
                    te_batch_durations.append(min_length)

                # store the weighted average results
                te_accuracy[str(te_snr)] = np.average(te_batch_accuracy, weights=te_batch_durations)

        te_accuracy['mean'] = np.mean(list(te_accuracy.values()))

        logging.info(json.dumps(te_accuracy, sort_keys=True, indent=4))
        F.write_data(filename=os.path.join(output_directory, f'test_results.txt'),
                     data=te_accuracy)

    return


evaluation()
logging.info('Completed testing Gating model (with {} hidden units and {} layers) to cluster mixtures by {}.'.format(
    args.hidden_size, args.num_layers, args.latent_space))
if args.disconnect:
    time.sleep(60)
    os.kill(os.getppid(), signal.SIGHUP)  # useful for closing tmux sessions

