# torch stuff
import torch
import torch.nn as nn

# others
from typing import Optional, Tuple



##################################################
class SiglipVisionConfig:

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_head=12,
        num_channels =3,
        image_size=224,
        patch_size=16,
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        num_image_tokens: int = None,
        **kwargs
    ):

        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_head
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.num_image_tokens = num_image_tokens
###########################################################


class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=config.patch_size,
            padding="valid"
        )
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )
        # no [CLS] embedding here.
    
    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        _, _, height, width = pixel_values.shape # <b, c, h, w>
        patch_embeds = self.patch_embedding(pixel_values)
        # patch_embeds -->  <b, embed_dim, patch_size, patch_size>
        embeddings = patch_embeds.flatten(2)
        #  embeddings --> <b, num_of_patches, embedding)
        embeddings = embeddings.transpose(1, 2)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        # pos_emb = self.position_embedding(self.position_ids)
        # print (f"position embeddings:  {pos_emb.shape}")
        return embeddings


class SiglipMLP(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class SiglipAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = self.config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        # all the linear layers for Q, K, V.
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        
    def forward(self, hidden_states: torch.Tensor
        )-> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, _ = hidden_states.size()
        
        # from here on attention operation starts.
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        # mm between Q and K, we split the original dim --> (num_heads, head_dim)
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim)
        query_states = query_states.transpose(1, 2) # swap seq_len axis with the self.num_heads so that we have following dim.
        # batch, self.num_heads, seq_len, self.head_dim
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim)
        key_states = key_states.transpose(1, 2)
        # batch, self.num_heads, seq_len, self.head_dim
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim)
        value_states = value_states.transpose(1, 2)
        attn_weights = (torch.matmul(query_states, key_states.transpose(2,3)) * self.scale)
        # the attn_weights is as follows:
        # query_states = (batch,  num_heads, seq_len, head_dim)
        # key_states = (batch, num_head, head_dim, seq_len)
        # so attn_weights dim ==> (batch, num_head, seq_len, seq_len)
        if attn_weights.size() != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError("attn weight size mismatch")
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states) # (b, num_head, seq_len, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous() # (b,  seq_len, num_head, head_dim)
        attn_output = attn_output.reshape(batch_size, seq_len, self.num_heads * self.head_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights
        # attn_output dim. derivation as follows:
        # attn_weights = (batch, num_head, seq_len, seq_len) 
        # value_states = (batch, num_head, seq_len, head_dim)
        # 
        # attn_weights = (torch.mat(query_states, key_states.transpose()))
        
class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_state --> normalized --> attn --> normalized again --> plus residual connection.
        residual = hidden_states # <b, no_of_patces, embed_dim>
        hidden_states = self.layer_norm1(hidden_states) 
        hidden_states, _ = self.self_attn(hidden_states=hidden_states)
        hidden_states += residual
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

class SiglipEncoder(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(config) for _ in  range(config.num_hidden_layers)]
        )
    
    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        # inputs_embeds: [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = inputs_embeds

        for encoder_layer in self.layers:
            # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
            hidden_states = encoder_layer(hidden_states)

        return hidden_states



class SiglipVisionTransformer(nn.Module):

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        # patch embeddings
        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config) 
        self.post_layernorm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        hidden_states = self.embeddings(x)
        last_hidden_state = self.encoder(inputs_embeds=hidden_states)
        last_hidden_state = self.post_layernorm(last_hidden_state)
        return last_hidden_state


class SiglipVisionModel(nn.Module):
    # SiglipVisonModel == SiglipVisionTransformer 
    # SiglipVisonTransformer = SiglipVisionEmbedding + SiglipEncoder
    # SiglipEncoder = SiglipAttention + SigLipMLP + SiglipNORM
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, x) -> tuple:
        # takes [B, C, H, W] returns [B, N, E]
        return self.vision_model(x)


config = SiglipVisionConfig()
print(config.__dict__)

print("-" * 20)

model = SiglipVisionModel(config)
# print (model)
input_ = torch.rand((2, 3, 224, 224))
out = model(input_)
# print (out.shape)

        