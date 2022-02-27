import torch
import numpy as np
class Spike(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        inputs,
        activation,
        dt=0.001,
        initial_state=None,
        spiking_aware_training=True,
        return_sequences=False,
        training=False,
    ):
        ctx.activation = activation
        ctx.return_sequences = return_sequences
        ctx.save_for_backward(inputs)

        if training and not spiking_aware_training:
            output = activation(inputs if return_sequences else inputs[:, -1])
            return output

        if initial_state is None:
            initial_state = torch.rand(
                inputs.shape[0], inputs.shape[2], dtype=inputs.dtype
            )
        inputs = inputs.type(initial_state.dtype) 
        voltage = initial_state
        all_spikes = []
        rates = activation(inputs) * dt
        arr = np.random.poisson(inputs.shape[1],inputs.shape[1])%inputs.shape[1]
        for i in arr:
            voltage += rates[:, i]
            n_spikes = torch.floor(voltage)
            voltage -= n_spikes
            if return_sequences:
                all_spikes.append(n_spikes)

        if return_sequences:
            output = torch.stack(all_spikes, dim=1)
        else:
            output = n_spikes

        output /= dt
        return output

    @staticmethod
    def backward(ctx, grad_output):
        inputs = ctx.saved_tensors[0]
        with torch.enable_grad():
            output = ctx.activation(inputs if ctx.return_sequences else inputs[:, -1])
            return (
                torch.autograd.grad(output, inputs, grad_outputs=grad_output)
                + (None,) * 7
            )

spike = Spike.apply
