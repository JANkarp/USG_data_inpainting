import torch.nn as nn
import torch
import numpy as np

#SIREN INR implementation

class SIREN(nn.Module):

    def __init__(self, in_features, out_features, hidden_features, hidden_layers = 1, first_omega = 30, hidden_omega = 30):
        super().__init__()

        net = []

        net.append(SineLayer(in_features, hidden_features, is_first=True, omega=first_omega))

        for _ in range(hidden_layers):
            net.append(SineLayer(hidden_features, hidden_features, is_first=False, omega=hidden_omega))

        final_linear = nn.Linear(hidden_features, out_features)

        init_weights(final_linear.weight, is_first=False, omega=hidden_omega)

        net.append(final_linear)

        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)

class SineLayer(nn.Module):

    def __init__(self, in_features, out_features, bias=True, is_first=False, omega = 30):
        super().__init__()
        self.omega = omega
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init_weights(self.linear.weight, is_first, omega)

    def forward(self, x):
        return torch.sin(self.omega * self.linear(x))


def init_weights(weight, is_first, omega = 1):

    in_features = weight.shape[1]

    with torch.no_grad():
        if is_first:
            bound = 1/in_features
        else:
            bound = np.sqrt(6/in_features) / omega

        weight.uniform_(-bound, bound)