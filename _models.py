import numpy as np
import torch
import torch.nn as nn

import _config as C
import _functions as F


class DenoisingNetwork(nn.Module):

    def __init__(self, hidden_size, num_layers):

        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.rnn = nn.LSTM(C.stft_features, hidden_size, num_layers, batch_first=True)
        self.dnn = nn.Linear(hidden_size, C.stft_features)
        F.initialize_weights(self)

    def forward(self, x):
        (batch_size, seq_len, num_features) = x.shape
        x = F.many_to_many(self.rnn(x))
        x = x.reshape(batch_size*seq_len, self.rnn.hidden_size)
        x = self.dnn(x)
        x = self.sigmoid(x)
        x = x.reshape(batch_size, seq_len, num_features)
        return x




class GatingNetwork(nn.Module):

    def __init__(self, hidden_size, num_layers, num_clusters):
        super().__init__()
        self.softmax = nn.Softmax(dim=1)
        self.rnn = nn.LSTM(C.stft_features, hidden_size, num_layers, batch_first=True)
        self.dnn = nn.Linear(hidden_size, num_clusters)
        F.initialize_weights(self)

    def forward(self, x, alpha=1):
        x = F.many_to_one(self.rnn(x))
        x = self.dnn(x)
        x = self.softmax(alpha*x)
        return x




class EnsembleNetwork(nn.Module):

    def __init__(self, filepath_gating, filepaths_denoising, g_hs, g_nl, s_hs, s_nl, ct='snr'):
        super().__init__()
        self.alpha = 1
        self.num_forwards = 0

        # load gating weights
        nc = (4 if ct == 'snr' else 2)
        self.gating = GatingNetwork(g_hs, g_nl, nc)
        self.gating.load_state_dict(torch.load(filepath_gating, map_location=torch.device('cpu')))

        # load specialist weights
        self.specialists = nn.ModuleList([])
        for filepath_denoising in filepaths_denoising:
            n = DenoisingNetwork(s_hs, s_nl)
            n.load_state_dict(torch.load(filepath_denoising, map_location=torch.device('cpu')))
            self.specialists.append(n)


    def anneal(self, strategy=0):
        if strategy == 1:
            self.alpha = np.power(1.0023, self.num_forwards)
        elif strategy == 2:
            self.alpha = 5 + F.logistic(self.num_forwards, L=5, k=0.01, x_o=500)
        elif strategy == 3:
            self.alpha = 5 * np.sin(0.005*self.num_forwards - 499.9) + 5.2
        elif strategy == 4:
            self.alpha = 10
        return


    def forward(self, x, strategy=0):

        if self.alpha < 10:
            self.anneal(strategy=strategy)
        # the gating network creates a selection mask which is a softmax
        # prediction for the contribution of each specialist
        p = self.gating(x, self.alpha)

        if self.training:
            self.num_forwards += 1
            p = p[..., None, None]

            # run each specialist network on the input, then scale the
            # specialist's mask by the contribution weighting, and finally merge
            # the specialist inferences
            o = torch.stack([
                p[:, k]*self.specialists[k](x) for k in range(len(self.specialists))
            ], dim=0).sum(dim=0)

        else:

            # we pick the specialist using 'hard' argmax
            k = int(p.sum(dim=0).argmax().item())

            # during test time feed forward, we only pass the input onto one
            # specialist for inference
            o = self.specialists[k](x)

        return o
