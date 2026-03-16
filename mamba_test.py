import torch
from transformers import MambaConfig
from transformers.models.mamba.modeling_mamba import MambaBlock

config = MambaConfig(hidden_size=256, state_size=16, num_hidden_layers=1)
block = MambaBlock(config, layer_idx=0)
x = torch.randn(2, 50, 256)
out = block(x)
print("x shape:", x.shape)
print("out[0] shape:", out[0].shape)
