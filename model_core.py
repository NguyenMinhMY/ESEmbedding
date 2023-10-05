import torch
import math
import torch.nn as nn
import torch.nn.init as init
from einops import repeat
from transformers import (
    AutoModel,
    AutoProcessor,
    Wav2Vec2Model,
)


class ESEmbedding(nn.Module):
    
    def __init__(self, config):
        super().__init__()

        self.sr = config['sampling_rate']
        self.device = config['device']

        processor = AutoProcessor.from_pretrained(config['pretrained'])
        pretrained_model = AutoModel.from_pretrained(config['pretrained'])
        hidden_size = pretrained_model.config.hidden_size
            
        self.processor = processor
        self.feature_extractor = pretrained_model.feature_extractor
        self.feature_projection = pretrained_model.feature_projection
        self.encoder = pretrained_model.encoder

        self.cls_token = nn.Parameter(torch.rand(1, 1, hidden_size))
        self.proj_out = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size)
        )
            
        # self._reinit_weight_of_the_last_layer(self.encoder.layers)
        
    def forward(self, signal):
        """
            signal: (B, T)
            ouput: (B, 1, D)
        """
        batch_size, _ = signal.shape
        
        processed_signal = self.processor(
            signal, 
            sampling_rate=self.sr, 
            return_tensors='pt'
        ).get('input_values').squeeze(0).to(self.device)

        features = self.feature_extractor(processed_signal)
        features = features.transpose(1, 2)
        hidden_states, features = self.feature_projection(features)
        
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = batch_size)
        hidden_states = torch.cat((cls_tokens, hidden_states), dim=1)
        
        hidden_states = self.encoder(hidden_states).last_hidden_state
        cls_emb = hidden_states[:, 0] # just pick the first feature, aka cls tokens
        output = self.proj_out(cls_emb)
        
        return output
    
    def _reinit_weight_of_the_last_layer(self, layers):
        for ith, layer in enumerate(layers):
            if ith == len(layers) - 1:
                self._init_weight(layer.attention.k_proj)
                self._init_weight(layer.attention.v_proj)
                self._init_weight(layer.attention.q_proj)
                self._init_weight(layer.attention.out_proj)
                if layer.layer_norm.elementwise_affine:
                    init.ones_(layer.layer_norm.weight)
                    init.zeros_(layer.layer_norm.bias)
                self._init_weight(layer.feed_forward.intermediate_dense)
                self._init_weight(layer.feed_forward.output_dense)
                if layer.final_layer_norm.elementwise_affine:
                    init.ones_(layer.final_layer_norm.weight)
                    init.zeros_(layer.final_layer_norm.bias)
    
    def _init_weight(self, layer):
        init.kaiming_normal_(layer.weight)
        if layer.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(layer.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(layer.bias, -bound, bound)