import torch
from torch import nn
from torchinfo import summary
from encoder import Encoder
from decoder import Decoder
from diffusion import UNet


encoder = Encoder()
decoder = Decoder()
diffusion = UNet()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder.to(device)

summary(encoder,  input_sizes=[(3, 256, 256), (4, 64, 64)])  

summary(decoder, input_sizes=[(4, 64, 64)])

summary(diffusion, input_sizes=[(4, 64, 64)])
