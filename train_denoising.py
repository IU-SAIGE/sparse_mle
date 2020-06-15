import argparse
import os
import time
import logging
import signal

import numpy as np
import torch

import _config as C
import _functions as F
import _models as M


#
# parse arguments
#
p = argparse.ArgumentParser()
p.add_argument('-l', '--learning_rate', type=float, required=True)
p.add_argument('-z', '--hidden_size', type=int, required=True)
p.add_argument('-n', '--num_layers', type=int, required=True)
p.add_argument('-s', '--specialization', choices=C.specializations, default=None)
p.add_argument('-d', '--device_id', default=F.get_gpu_next_device())
p.add_argument('--disconnect', action='store_true')
args = p.parse_args()


#
# create results directory
#
output_directory = C.results_directory + '/Denoising/' \
                   f'/lr{args.learning_rate:.0e}' \
                   f'/{F.fmt_specialty(args.specialization)}' \
                   f'/{args.hidden_size:04}x{args.num_layers}/'
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
(tr_utterances, tr_noises) = filepaths[0:2]
(vl_utterances, vl_noises) = filepaths[2:4]

(tr_snr, tr_gender) = (C.snr_all, C.gender_all)
args.latent_space = None
if args.specialization is not None:
    if args.specialization.isnumeric():
        assert int(args.specialization) in set(C.snr_all)
        args.latent_space = 'snr'
        tr_snr = int(args.specialization)
    elif args.specialization.isalpha():
        assert str(args.specialization) in set(C.gender_all)
        args.latent_space = 'gender'
        tr_utterances = F.filter_by_gender(
            tr_utterances,
            str(args.specialization)
        )


#
# experiment
#

def experiment():

    #
    # initialize network
    #
    np.random.seed(0)
    torch.manual_seed(0)

    network = M.DenoisingNetwork(
        args.hidden_size,
        args.num_layers
    ).to(device=args.device_id)

    network_params = F.count_parameters(network)

    optimizer = torch.optim.Adam(
        params=network.parameters(),
        lr=args.learning_rate,
    )

    criterion = F.loss_sisdr

    F.write_data(filename=os.path.join(output_directory, 'num_parameters.txt'),
                 data=network_params)

    with torch.cuda.device(args.device_id):
        torch.cuda.empty_cache()


    #
    # log experiment configuration
    #
    os.system('cls' if os.name == 'nt' else 'clear')
    logging.info(f'Training Denoising network'+(f' specializing in {F.fmt_specialty(args.specialization)} mixtures' if args.specialization else '')+'...')
    logging.info(f'\u2022 {args.hidden_size} hidden units')
    logging.info(f'\u2022 {args.num_layers} layers')
    logging.info(f'\u2022 {network_params} learnable parameters')
    logging.info(f'\u2022 {args.learning_rate:.3e} learning rate')
    logging.info(f'Results will be saved in "{output_directory}".')
    logging.info(f'Using GPU device {args.device_id}...')


    #
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
                mixture_snr=tr_snr,
                device=args.device_id,
            )
            M_hat = network(batch.X_mag)
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
        if sisdr_batch['mean' if args.latent_space != 'snr' else tr_snr] > sisdr_best:
            sisdr_best = sisdr_batch['mean' if args.latent_space != 'snr' else tr_snr]
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
        iteration += 1

    return


# attempt running the experiment up to N times
for attempt in range(C.num_attempts):
    try:

        ts_start = int(round(time.time()))
        experiment()

    except RuntimeError as e:

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
