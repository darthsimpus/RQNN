import numpy as np
import torch

from random_equivalent.methods import spike


class SpikingActivation(torch.nn.Module): 

    def __init__(
        self,
        activation,
        dt=0.001,
        initial_state=None,
        spiking_aware_training=True,
        return_sequences=True,
    ):
        super().__init__()

        self.activation = activation
        self.initial_state = initial_state
        self.dt = dt
        self.spiking_aware_training = spiking_aware_training
        self.return_sequences = return_sequences

    def forward(self, inputs):

        return spike(
            inputs,
            self.activation,
            self.dt,
            self.initial_state,
            self.spiking_aware_training,
            self.return_sequences,
            self.training,
        )

    def forward(self, inputs):
        if self.training and not self.apply_during_training:
            return inputs if self.return_sequences else inputs[:, -1]

        level = self.level_var
        smoothing = torch.sigmoid(self.smoothing_var)

        # cast inputs to module type
        inputs = inputs.type(self.smoothing_var.dtype)

        all_levels = []
        for i in range(inputs.shape[1]):
            level = (1 - smoothing) * inputs[:, i] + smoothing * level
            if self.return_sequences:
                all_levels.append(level)

        if self.return_sequences:
            return torch.stack(all_levels, dim=1)
        else:
            return level


class TemporalAvgPool(torch.nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, inputs):
        return torch.mean(inputs, dim=self.dim)
