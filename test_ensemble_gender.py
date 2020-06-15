import argparse
import os
import time
import logging
import signal
import re

import numpy as np
import torch

from collections import namedtuple
from math import ceil

import _functions as F

from _config import hop_size
from _config import gender_all
from _config import snr_all
from _models import GatingNetwork
from _models import SpecialistNetwork


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
model_name = 'Ensemble'
cluster_type = 'gender'

m = re.search(r'\/(\d+)x(\d+)\/', args.state_dict_file_gating)
hidden_size_gating = int(m[1])
num_layers_gating = int(m[2])
architecture_gating = f'{m[1]}x{m[2]}'

m = re.search(r'\/(\d+)x(\d+)\/', args.state_dict_file_specialist)
hidden_size_specialist = int(m[1])
num_layers_specialist = int(m[2])
architecture_specialist = f'{m[1]}x{m[2]}'

output_directory = './results-2020_04_24/{}/{}/g{}/s{}/'.format(
    model_name, cluster_type, architecture_gating, architecture_specialist)
output_directory = os.path.abspath(os.path.expanduser(output_directory))
os.makedirs(output_directory, exist_ok=True)


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
(vl_utterances, vl_noises) = filepaths[2:4]
(te_utterances, te_noises) = filepaths[4:6]
(vl_batch_size, te_batch_size) = (1000, 1000)


#
# evaluation
#

def evaluation():
    with torch.no_grad():

        sum_num_params = 0

        #
        # initialize gating network
        #
        gating = GatingNetwork(hidden_size_gating, num_layers_gating, len(gender_all)).to(device=args.device_id)
        gating.load_state_dict(torch.load(
            args.state_dict_file_gating, map_location=torch.device(args.device_id))
        )
        gating.eval()
        sum_num_params += F.count_parameters(gating)


        #
        # initialize specialist networks (as a hashed list of networks)
        #
        specialists = {
            i: SpecialistNetwork(hidden_size_specialist, num_layers_specialist).to(device=args.device_id)
            for i in range(len(gender_all))
        }
        for i in range(len(gender_all)):
            assert (re.search(r'gender\_[MF]', args.state_dict_file_specialist) is not None)
            filepath = re.sub(r'gender\_[MF]', F.fmt_gender(gender_all[i]), args.state_dict_file_specialist)
            specialists[i].load_state_dict(torch.load(
                filepath, map_location=torch.device(args.device_id))
            )
            specialists[i].eval()
            sum_num_params += F.count_parameters(specialists[i])


        F.write_data(filename=os.path.join(output_directory, 'num_parameters.txt'),
                     data=sum_num_params)
        with torch.cuda.device(args.device_id):
            torch.cuda.empty_cache()


        #
        # log experiment configuration
        #
        logging.info('All results will be stored in "{}".'.format(
            output_directory))
        logging.info('Testing {} model (with Gating architecture {} and Specialist architecture {}) to denoise {} gendered mixtures...'.format(
            model_name, architecture_gating, architecture_specialist, gender_all))
        logging.info('Using GPU device {}...'.format(
            args.device_id))


        fields = ['snr_val','num_mixtures','sdr','sisdr','mse','bce','accuracy']

        #
        # validation
        #
        results_validation = []
        np.random.seed(0)
        torch.manual_seed(0)
        for gender_val in gender_all:

            # construct a batch
            batch = F.generate_batch(
                vl_utterances, vl_noises,
                batch_size=vl_batch_size,
                gender=gender_val,
                device=args.device_id,
            )
            Y = batch.index_gender

            # compute batch-wise specialist probabilities
            Y_hat = gating(batch.X_mag)

            # pick the best specialist to apply to the whole batch (based on batch probabilities sum)
            k = int(Y_hat.sum(dim=0).argmax().item())

            # apply the best specialist to the entire batch
            M_hat = specialists[k](batch.X_mag)
            s_hat = F.istft(batch.X, mask=M_hat)

            results_validation.append([
                gender_val,
                vl_batch_size,
                float(F.calculate_sdr(batch.s, s_hat, offset=batch.actual_sdr).mean().item()),
                float(F.calculate_sisdr(batch.s, s_hat, offset=batch.actual_sisdr).mean().item()),
                float(F.calculate_mse(batch.M, M_hat).item()),
                float(F.calculate_bce(batch.M, M_hat).item()),
                float(F.calculate_accuracy(Y, Y_hat)),
            ])
            status = (
                f'Validation Data (Gender={gender_val}) -- ' + \
                f'SDR: {results_validation[-1][2]:>6.3f} dB, ' + \
                f'\033[33mSISDR: {results_validation[-1][3]:>6.3f} dB\033[39m, ' + \
                f'MSE: {results_validation[-1][4]:>6.3f}, ' + \
                f'BCE: {results_validation[-1][5]:>6.3f}, ' + \
                f'Accuracy: {results_validation[-1][6]:>6.3f}'
            )
            logging.info(status)
        F.write_table(filename=os.path.join(output_directory, f'validation_results.txt'),
                      table_data=results_validation, headers=fields)


        #
        # testing
        #
        results_testing = []

        for gender_val in gender_all:
            np.random.seed(0)
            torch.manual_seed(0)
            te_utterances_filtered = te_utterances[np.array([(F.get_gender(row) in gender_val) for row in te_utterances])]
            files_speech = np.random.choice(te_utterances_filtered, size=te_batch_size)
            files_noise = np.random.choice(te_noises, size=te_batch_size)
            te_m_durations = list()
            te_m_sdr = list()
            te_m_sisdr = list()
            te_m_mse = list()
            te_m_bce = list()
            te_m_accuracy = list()
            for (i, fs, fn) in zip(range(te_batch_size), files_speech, files_noise):

                source = F.load_audio(fs, duration=None, random_offset=False, device=args.device_id)
                noise = F.load_audio(fn, duration=None, random_offset=False, device=args.device_id)
                min_length = min(len(source), len(noise))
                stft_frames = ceil(min_length/hop_size)
                source = source[:min_length]
                noise = noise[:min_length]

                (x, s, n) = F.mix_signals(source, noise, snr_db=snr_all)
                (S, S_mag) = F.stft(s)
                (N, N_mag) = F.stft(n)
                (X, X_mag) = F.stft(x)
                (M) = F.calculate_masking_target(S_mag, N_mag)

                X = X.permute(1, 0, 2)[:stft_frames] # (seq_len, num_features, channel)
                S_mag = S_mag.permute(1, 0)[:stft_frames]  # (seq_len, num_features)
                N_mag = N_mag.permute(1, 0)[:stft_frames]  # (seq_len, num_features)
                X_mag = X_mag.permute(1, 0)[:stft_frames]  # (seq_len, num_features)
                M = M.permute(1, 0)[:stft_frames]  # (seq_len, num_features)

                actual_sdr = float(F.calculate_sdr(s, x).item())
                actual_sisdr = float(F.calculate_sisdr(s, x).item())

                gender_index = int(F.get_gender(fs)=='F')
                Y = torch.zeros(1, len(gender_all), device=args.device_id)
                Y[..., gender_index] = 1

                # add a fake batch axis to everything
                x = torch.unsqueeze(x, dim=0)
                s = torch.unsqueeze(s, dim=0)
                n = torch.unsqueeze(n, dim=0)
                S = torch.unsqueeze(S, dim=0)
                S_mag = torch.unsqueeze(S_mag, dim=0)
                N = torch.unsqueeze(N, dim=0)
                N_mag = torch.unsqueeze(N_mag, dim=0)
                X = torch.unsqueeze(X, dim=0)
                X_mag = torch.unsqueeze(X_mag, dim=0)
                M = torch.unsqueeze(M, dim=0)

                # compute batch-wise specialist probabilities
                Y_hat = gating(X_mag)

                # pick the best specialist to apply to the whole batch (based on batch probabilities sum)
                k = int(Y_hat.sum(dim=0).argmax().item())

                # apply the best specialist to the entire batch
                M_hat = specialists[k](X_mag)
                s_hat = F.istft(X, mask=M_hat)

                te_m_sdr.append(F.calculate_sdr(s, s_hat, offset=actual_sdr).mean().item())
                te_m_sisdr.append(F.calculate_sisdr(s, s_hat, offset=actual_sisdr).mean().item())
                te_m_mse.append(F.calculate_mse(M, M_hat).item())
                te_m_bce.append(F.calculate_bce(M, M_hat).item())
                te_m_accuracy.append(float(torch.prod(Y==torch.round(Y_hat), dim=-1).sum().item()/float(len(Y))))
                te_m_durations.append(min_length)

            # store the weighted average results
            results_testing.append([
                gender_val,
                te_batch_size,
                np.average(te_m_sdr, weights=te_m_durations),
                np.average(te_m_sisdr, weights=te_m_durations),
                np.average(te_m_mse, weights=te_m_durations),
                np.average(te_m_bce, weights=te_m_durations),
                np.average(te_m_accuracy, weights=te_m_durations),
            ])
            status = (
                f'Test Data (Gender={gender_val}) -- ' + \
                f'SDR: {results_testing[-1][2]:>6.3f} dB, ' + \
                f'\033[33mSISDR: {results_testing[-1][3]:>6.3f} dB\033[39m, ' + \
                f'MSE: {results_testing[-1][4]:>6.3f}, ' + \
                f'BCE: {results_testing[-1][5]:>6.3f}, ' + \
                f'Accuracy: {results_testing[-1][6]:>6.3f}'
            )
            logging.info(status)
        F.write_table(filename=os.path.join(output_directory, f'test_results.txt'),
                      table_data=results_testing, headers=fields)
    return



ts_start = int(round(time.time()))
evaluation()
ts_end = int(round(time.time())) - ts_start
logging.info('Completed testing {} model (with Gating architecture {} and Specialist architecture {}) to denoise {} dB mixtures...'.format(
    model_name, architecture_gating, architecture_specialist, snr_all))
logging.info('Exiting `test_ensemble` after {} of execution.'.format(
    f'{int(ts_end/60)} minutes'))
if args.disconnect:
    os.kill(os.getppid(), signal.SIGHUP)  # useful for closing tmux sessions

