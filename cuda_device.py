import torch

CUDA_DEVICE = 'gpu' if torch.cuda.is_available() else 'cpu'
