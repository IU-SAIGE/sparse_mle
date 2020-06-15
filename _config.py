from math import ceil
import numpy as np
import yaml

sample_rate = 16000  # Hz
lookback = 1  # second

fft_size = 1024
hop_size = 256

num_samples = int(lookback*sample_rate)
stft_features = int(fft_size//2+1)
stft_frames = ceil(num_samples/hop_size)

snr_all = np.array([-5, 0, 5, 10])
gender_all = np.array(['M', 'F'])

specializations = np.concatenate([
	snr_all,
	gender_all
])
num_clusters = {
	'snr': len(snr_all),
	'gender': len(gender_all),
}

stopping_criteria = lambda a, b: (abs(a - b) > 20)

with open('genders.yaml', 'r') as fp:
	gender_map: dict = yaml.safe_load(fp)


tr_batch_size = 100
vl_batch_size = 1000
te_batch_size = 1000
num_attempts = 20

results_directory = './results-2020_05_04/'