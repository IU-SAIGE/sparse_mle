import argparse
import os
import time
import logging
import signal
import json
import re
import soundfile as sf
import matplotlib.pyplot as plt

import numpy as np
import torch

from math import ceil

import _config as C
import _functions as F

from _models import DenoisingNetwork
from _models import GatingNetwork
from _models import EnsembleNetwork


def im_write(filename, data):
    plt.subplots(nrows=1, ncols=1, dpi=300)
    plt.imshow(data, aspect='auto', origin='lower')
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight', transparent=True, pad_inches=0)
    plt.close('all')
    return


def load_dirty_json(dirty_json):
    regex_replace = [(r"([ \{,:\[])(u)?'([^']+)'", r'\1"\3"'), (r" False([, \}\]])", r' false\1'), (r" True([, \}\]])", r' true\1')]
    for r, s in regex_replace:
        dirty_json = re.sub(r, s, dirty_json)
    clean_json = json.loads(dirty_json)
    return clean_json


#
# parse arguments
#
p = argparse.ArgumentParser()
p.add_argument('-p', '--model_path', type=str, required=True)
p.add_argument('-d', '--device_id', default=F.get_gpu_next_device())
p.add_argument('--disconnect', action='store_true')
args = p.parse_args()
assert os.path.exists(args.model_path)
output_directory = os.path.dirname(args.model_path) + '/examples'
os.makedirs(output_directory, exist_ok=True)


#
# define logger
#
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [PID %(process)d] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(os.path.join(output_directory, 'examples.log')),
        logging.StreamHandler(),
    ],
)


#
# load audio filepaths
#
filepaths = np.load('filepaths.npy', allow_pickle=True)
(te_utterances, te_noises) = filepaths[4:6]
num_examples = 10
offset = 0
male_examples = [te_utterances[i] for i in range(len(te_utterances)) if F.get_gender(te_utterances[i]) == 'M'][offset:offset+num_examples]
female_examples = [te_utterances[i] for i in range(len(te_utterances)) if F.get_gender(te_utterances[i]) == 'F'][offset:offset+num_examples]


with torch.no_grad():

    #
    # parse model parameters from filepath
    #
    speciality = ''
    match = re.search(r'(snr|gen*)', args.model_path)
    args.latent_space = 'snr'
    if match:
        if ('gen' in str(match[1])):
            args.latent_space = 'gender'

    if re.search(r'[Ee]nsemble_with(?:out)?_FT', args.model_path):
        match = re.search(r'\/g(\d+)x(\d+)', args.model_path)
        hidden_size_gating = int(match[1])
        num_layers_gating = int(match[2])
        architecture_gating = f'{match[1]}x{match[2]}'

        match = re.search(r'\/s(\d+)x(\d+)', args.model_path)
        hidden_size_specialist = int(match[1])
        num_layers_specialist = int(match[2])
        architecture_specialist = f'{match[1]}x{match[2]}'

        files_gating = json.load(open(os.path.dirname(args.model_path) + '/files_gating.txt', 'r'))
        files_specialists = load_dirty_json(open(os.path.dirname(args.model_path) + '/files_specialist.txt', 'r').read().strip('"'))
        if not isinstance(files_specialists, list):
            if args.latent_space == 'snr':
                files_specialists = [
                    re.sub(r'snr_[np]\d+', F.fmt_snr(i), files_specialists)
                    for i in C.snr_all
                ]
            elif args.latent_space == 'gender':
                files_specialists = [
                    re.sub(r'gen_[MF]', 'gen_M', files_specialists),
                    re.sub(r'gen_[MF]', 'gen_F', files_specialists),
                ]
        np.random.seed(0)
        torch.manual_seed(0)

        network = EnsembleNetwork(
            filepath_gating=files_gating,
            filepaths_denoising=files_specialists,
            g_hs=hidden_size_gating,
            g_nl=num_layers_gating,
            s_hs=hidden_size_specialist,
            s_nl=num_layers_specialist,
            ct=args.latent_space,
        ).to(device=args.device_id)

    elif re.search(r'[Bb]aseline', args.model_path):
        args.latent_space = 'all'
        match = re.search(r'(\d+)x(\d)', args.model_path)
        hidden_size = int(match[1])
        num_layers = int(match[2])

        network = DenoisingNetwork(
            hidden_size,
            num_layers
        ).to(device=args.device_id)
        network.load_state_dict(torch.load(
            os.path.dirname(args.model_path) + '/model.pt',
            map_location=torch.device(args.device_id),
        ), strict=True)

    elif re.search(r'([Ss]pecialist|[Dd]enoising)', args.model_path):
        match = re.search(r'(\d+)x(\d)', args.model_path)
        hidden_size = int(match[1])
        num_layers = int(match[2])
        match = re.search(r'[A-z]+\_([pn]\d\d|[MF])', args.model_path)
        speciality = str(match[0])

        network = DenoisingNetwork(
            hidden_size,
            num_layers
        ).to(device=args.device_id)
        network.load_state_dict(torch.load(
            os.path.dirname(args.model_path) + '/model.pt',
            map_location=torch.device(args.device_id),
        ), strict=True)


    network.eval()
    print(args.latent_space, speciality, network)

    #
    # evaluate
    #
    speaker_examples = te_utterances[offset:offset+num_examples]
    noise_examples = te_noises[offset:offset+num_examples]
    te_snrs = C.snr_all
    if '_M' in speciality:
        speaker_examples = male_examples[offset:offset+num_examples]
    elif '_F' in speciality:
        speaker_examples = female_examples[offset:offset+num_examples]
    elif 'n05' in speciality:
        te_snrs = [-5]
    elif 'p00' in speciality:
        te_snrs = [0]
    elif 'p05' in speciality:
        te_snrs = [5]
    elif 'p10' in speciality:
        te_snrs = [10]


    for (i, fs, fn) in zip(range(offset, offset+len(speaker_examples)), speaker_examples, noise_examples):
        if not ((i == 4) or (i == 9)):
            continue
        for mixture_snr in te_snrs:
            if not (mixture_snr == -5):
                continue

            # mix the signals up
            source = F.load_audio(fs, device=args.device_id, random_offset=False, duration=None)
            noise = F.load_audio(fn, device=args.device_id, random_offset=False, duration=None)
            min_length = min(len(source), len(noise))
            (x, s, n) = F.mix_signals(source[:min_length], noise[:min_length], snr_db=mixture_snr)
            (S, S_mag) = F.stft(s)
            (N, N_mag) = F.stft(n)
            (X, X_mag) = F.stft(x)
            (M) = F.calculate_masking_target(S_mag, N_mag)
            X = X.permute(1, 0, 2)
            S_mag = S_mag.permute(1, 0)
            N_mag = N_mag.permute(1, 0)
            X_mag = X_mag.permute(1, 0)
            M = M.permute(1, 0)
            actual_sdr = float(F.calculate_sdr(s, x).item())
            actual_sisdr = float(F.calculate_sisdr(s, x).item())

            # inference
            M = network(X_mag.unsqueeze(0))
            y = F.istft(X.unsqueeze(0), mask=M)
            (Y, Y_mag) = F.stft(y)
            y = y.squeeze()
            output_sdr = float(F.calculate_sdr(s, y).item())
            output_sisdr = float(F.calculate_sisdr(s, y).item())

            # convert everything back into numpy types
            max_amplitude = 1e-30 + 1.15 * float(max(s.max(), n.max(), x.max(), y.max()))
            s = s.detach().cpu().numpy() / max_amplitude
            n = n.detach().cpu().numpy() / max_amplitude
            x = x.detach().cpu().numpy() / max_amplitude
            y = y.detach().cpu().numpy() / max_amplitude
            S_mag = S_mag.detach().cpu().numpy().transpose(1,0)
            N_mag = N_mag.detach().cpu().numpy().transpose(1,0)
            X_mag = X_mag.detach().cpu().numpy().transpose(1,0)
            Y_mag = Y_mag.detach().cpu().numpy().squeeze()
            M = M.detach().cpu().numpy().squeeze().transpose(1,0)

            # save everything to file
            sf.write(os.path.join(output_directory, f'{i:02}_{F.fmt_snr(mixture_snr)}_1_source.wav'), s, 16000)
            sf.write(os.path.join(output_directory, f'{i:02}_{F.fmt_snr(mixture_snr)}_2_noise.wav'), n, 16000)
            sf.write(os.path.join(output_directory, f'{i:02}_{F.fmt_snr(mixture_snr)}_3_mixture.wav'), x, 16000)
            sf.write(os.path.join(output_directory, f'{i:02}_{F.fmt_snr(mixture_snr)}_4_reconst.wav'), y, 16000)
            im_write(os.path.join(output_directory, f'{i:02}_{F.fmt_snr(mixture_snr)}_1_source.png'), np.log10(1+np.abs(S_mag)))
            im_write(os.path.join(output_directory, f'{i:02}_{F.fmt_snr(mixture_snr)}_2_noise.png'), np.log10(1+np.abs(N_mag)))
            im_write(os.path.join(output_directory, f'{i:02}_{F.fmt_snr(mixture_snr)}_3_mixture.png'), np.log10(1+np.abs(X_mag)))
            im_write(os.path.join(output_directory, f'{i:02}_{F.fmt_snr(mixture_snr)}_4_reconst.png'), np.log10(1+np.abs(Y_mag)))
            im_write(os.path.join(output_directory, f'{i:02}_{F.fmt_snr(mixture_snr)}_5_mask.png'), M)
            F.write_data(os.path.join(output_directory, f'{i:02}_{F.fmt_snr(mixture_snr)}_6_sdr_input.txt'), actual_sdr)
            F.write_data(os.path.join(output_directory, f'{i:02}_{F.fmt_snr(mixture_snr)}_6_sisdr_input.txt'), actual_sisdr)
            F.write_data(os.path.join(output_directory, f'{i:02}_{F.fmt_snr(mixture_snr)}_6_sdr_improvement.txt'), output_sdr-actual_sdr)
            F.write_data(os.path.join(output_directory, f'{i:02}_{F.fmt_snr(mixture_snr)}_6_sisdr_improvement.txt'), output_sisdr-actual_sisdr)
            logging.info(f'Wrote example #{i+1} to folder "{output_directory}".')
