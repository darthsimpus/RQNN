# Random Quantum Neural Network(RQNN)
This repository implements the RQNN model.

Author's code: https://github.com/darthsimpus/RQNN

## Dependecies
- Python 3.7
- PyTorch 1.7.1
- numpy

## Dataset

RQNN uses noisy versions of the datasets summarized below. Noisy versions were constructed by adding Salt & Pepper, Gaussian , Rayleigh , Uniform and Perlin noises of varying degrees of intensity.

Dataset summary:

| Dataset | #Classes | #Training & Validation Size  | #Testing Size  |
| :-: | :-: | :-: | :-: |
| MNIST | 4 | 954 | 7,073 |
| Fashion-MNIST | 4 | 954 | 7,172 |
| K-MNIST | 4 | 954 | 5,512 |
