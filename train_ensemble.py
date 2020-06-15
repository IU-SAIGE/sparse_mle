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
p.add_argument('-l', '--learning_rate', type=float, required=True)
p.add_argument('-sg', '--state_dict_file_gating', type=str, required=True)
p.add_argument('-ss', '--state_dict_file_specialist', type=str, required=True)
p.add_argument('-d', '--device_id', default=F.get_gpu_next_device())
p.add_argument('-a', '--softmax_annealing', type=int, choices=[0,1,2,3,4])
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

output_directory = C.results_directory + '/Ensemble_with_FT/' \
                   f'/lr{args.learning_rate:.0e}' \
                   f'/{args.latent_space}' \
                   f'/a{args.softmax_annealing}' \
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
        logging.FileHandler(os.path.join(output_directory, 'training.log')),
        logging.StreamHandler(),
    ],
)


#
# load audio filepaths
#
filepaths = np.load('filepaths.npy', allow_pickle=True)
(tr_utterances, tr_noises) = filepaths[0:2]
(vl_utterances, vl_noises) = filepaths[2:4]
(te_utterances, te_noises) = filepaths[4:6]


#
# experiment
#

def experiment():

    #
    # ensure reproducibility
    #
    np.random.seed(0)
    torch.manual_seed(0)

    #
    # initialize network
    #
    network = EnsembleNetwork(
        filepath_gating=file_gating,
        filepaths_denoising=files_specialists,
        g_hs=hidden_size_gating,
        g_nl=num_layers_gating,
        s_hs=hidden_size_specialist,
        s_nl=num_layers_specialist,
        ct=args.latent_space,
    ).to(device=args.device_id)

    optimizer = torch.optim.Adam(
        params=network.parameters(),
        lr=args.learning_rate,
    )

    network_params = F.count_parameters(network.gating) + F.count_parameters(network.specialists[0])

    F.write_data(filename=os.path.join(output_directory, 'num_parameters.txt'),
                 data=network_params)
    F.write_data(filename=os.path.join(output_directory, 'files_gating.txt'),
                 data=file_gating)
    F.write_data(filename=os.path.join(output_directory, 'files_specialist.txt'),
                 data=files_specialists)
    with torch.cuda.device(args.device_id):
        torch.cuda.empty_cache()


    #
    # log experiment configuration
    #
    os.system('cls' if os.name == 'nt' else 'clear')
    logging.info(f'Training Ensemble network composed of {args.latent_space} specialists...')
    logging.info(f'\u2022 {architecture_gating} gating architecture')
    logging.info(f'\u2022 {architecture_specialist} specialist architecture')
    logging.info(f'\u2022 Softmax annealing strategy = {args.softmax_annealing if args.softmax_annealing else None}')
    logging.info(f'\u2022 {network_params} learnable parameters')
    logging.info(f'\u2022 {args.learning_rate:.3e} learning rate')
    logging.info(f'Results will be saved in "{output_directory}".')
    logging.info(f'Using GPU device {args.device_id}...')


    # softmax_annealing
    # experiment loop
    #
    (iteration, iteration_best) = (0, 0)
    sisdr_best = 0

    while not C.stopping_criteria(iteration, iteration_best):

        network.train()
        np.random.seed(iteration)
        torch.manual_seed(iteration)

        # training
        for batch_index in range(100):

            # forward propagation
            batch = F.generate_batch(
                np.random.choice(tr_utterances, size=C.tr_batch_size),
                np.random.choice(tr_noises, size=C.tr_batch_size),
                device=args.device_id,
            )
            M_hat = network(batch.X_mag, args.softmax_annealing)
            s_hat = F.istft(batch.X, mask=M_hat)

            # backward propagation
            optimizer.zero_grad()
            F.loss_sisdr(batch.s, s_hat).backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1e-4)
            optimizer.step()

        network.eval()
        np.random.seed(0)
        torch.manual_seed(0)

        # validation
        with torch.no_grad():

            if args.latent_space == 'gender':

                sisdr_batch = {k: 0 for k in C.gender_all}
                for vl_gender in C.gender_all:

                    vl_filtered_files = F.filter_by_gender(
                        vl_utterances,
                        vl_gender
                    )
                    batch = F.generate_batch(
                        np.random.choice(vl_filtered_files, size=C.vl_batch_size),
                        np.random.choice(vl_noises, size=C.vl_batch_size),
                        device=args.device_id,
                    )
                    M_hat = network(batch.X_mag)
                    s_hat = F.istft(batch.X, mask=M_hat)
                    sisdr_batch[vl_gender] = float(F.calculate_sisdr(batch.s, s_hat, offset=batch.actual_sisdr).mean().item())

            else:

                sisdr_batch = {k: 0 for k in C.snr_all}
                for vl_snr in C.snr_all:

                    batch = F.generate_batch(
                        np.random.choice(vl_utterances, size=C.vl_batch_size),
                        np.random.choice(vl_noises, size=C.vl_batch_size),
                        mixture_snr=vl_snr,
                        device=args.device_id,
                    )
                    M_hat = network(batch.X_mag)
                    s_hat = F.istft(batch.X, mask=M_hat)
                    sisdr_batch[vl_snr] = float(F.calculate_sisdr(batch.s, s_hat, offset=batch.actual_sisdr).mean().item())

        sisdr_batch['mean'] = np.mean(list(sisdr_batch.values()))


        # print results
        if sisdr_batch['mean'] > sisdr_best:
            sisdr_best = sisdr_batch['mean']
            iteration_best = iteration

            F.write_data(
                filename=os.path.join(output_directory, 'validation_sisdr.txt'),
                data=f'{sisdr_best:%}'
            )
            torch.save(network.state_dict(), os.path.join(output_directory, 'model.pt'))
            checkmark = ' | \033[32m\u2714\033[39m'
        else:
            checkmark = ''

        status = ''
        for (k, v) in sisdr_batch.items():
            status += f'\033[33m{k}: {v:>6.3f} dB\033[39m, '
        ts_end = int(round(time.time())) - ts_start
        status += f'Time Elapsed: {int(ts_end/60)} minutes' + checkmark
        logging.info(status)
        logging.info(f'Network # of forwards: {network.num_forwards} \t Alpha: {network.alpha}')
        iteration += 1

    return os.path.join(output_directory, 'model.pt')



def evaluation(model_path):
    with torch.no_grad():

        #
        # initialize network
        #
        np.random.seed(0)
        torch.manual_seed(0)

        network = EnsembleNetwork(
            filepath_gating=file_gating,
            filepaths_denoising=files_specialists,
            g_hs=hidden_size_gating,
            g_nl=num_layers_gating,
            s_hs=hidden_size_specialist,
            s_nl=num_layers_specialist,
            ct=args.latent_space,
        ).to(device=args.device_id)

        network.load_state_dict(torch.load(
            model_path,
            map_location=torch.device(args.device_id),
        ), strict=True)

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




# attempt running the experiment up to N times
for attempt in range(C.num_attempts):
    try:

        ts_start = int(round(time.time()))
        model_path = experiment()
        evaluation(model_path)

    except RuntimeError as e:

        np.random.seed(None)
        delay = np.random.randint(60) + 20

        if ('out of memory' in str(e)):
            logging.info(f'Out of resources on GPU {args.device_id}, will retry on another GPU in {delay} min...')
        elif (args.device_id < 0):
            logging.info(f'No available GPUs right now, will re-check in {delay} min...')
        else:
            raise e

        time.sleep(60*delay)
        args.device_id = F.get_gpu_next_device()

    else:

        logging.info('Completed training after {} of execution.'.format(
            f'{int(((round(time.time())) - ts_start)/60)} minutes'))
        if args.disconnect:
            os.kill(os.getppid(), signal.SIGHUP)  # useful for closing tmux sessions
        break
