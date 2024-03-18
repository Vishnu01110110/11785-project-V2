import torch
from torch import nn
from torchinfo import summary
from encoder import Encoder
from encoder2 import VAE_Encoder


model = Encoder()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# dummies
# input_tensor = torch.rand(1, 3, 256, 256).to(device)
# noise_shape = (1, 4, 32, 32)  
# noise = torch.rand(noise_shape).to(device)

summary(model,  input_sizes=[(3, 256, 256), (4, 64, 64)])  
