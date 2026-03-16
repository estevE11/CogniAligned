import sys
import torch
import torch.nn as nn
from transformers import MambaConfig, MambaModel

class Config:
    def __init__(self):
        self.hidden_size = 768
        self.n_layers = 1
        self.n_heads = 12
        self.intermediate_size = 3072
        self.dropout = 0.3
        self.num_classes = 1
        self.hidden_mlp_size = 256
        self.model_name = "distil_wav2vec2"
        self.fusion = "crossgated"
        self.audio_model = "wav2vec2"
        self.pooling = "mean"

config = Config()

sys.path.append("modules")
from model import CrossAttentionTransformerEncoder, MambaFusionEncoder

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

cross = CrossAttentionTransformerEncoder(config)
config.fusion = "mambaconcat"
mamba = MambaFusionEncoder(config)

print(f"CrossAttention parameters: {count_parameters(cross):,}")
print(f"Mamba parameters: {count_parameters(mamba):,}")
