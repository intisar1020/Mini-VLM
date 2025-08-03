import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


class SiglipVisionConfig:
    """A dummy config for SiglipVisionModel."""
    def __init__(self, **kwargs):
        self.hidden_size = kwargs.get("hidden_size", 1024)
        self.image_size = kwargs.get("image_size", 224)
        self.patch_size = kwargs.get("patch_size", 14)
        self.projection_dim = kwargs.get("projection_dim", 2048)
        self.intermediate_size = kwargs.get("intermediate_size", 4096)
        self.num_attention_heads = kwargs.get("num_attention_heads", 16)
        self.num_hidden_layers = kwargs.get("num_hidden_layers", 24)

class SiglipVisionModel(nn.Module):
    """A dummy SiglipVisionModel to show the structure."""
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.patch_embed = nn.Conv2d(3, config.hidden_size, kernel_size=config.patch_size, stride=config.patch_size)
        self.encoder = nn.Sequential(*[nn.Linear(config.hidden_size, config.hidden_size) for _ in range(config.num_hidden_layers)])
        self.pooler = nn.AdaptiveAvgPool1d(1)

    def forward(self, pixel_values, **kwargs):
       return torch.randn(pixel_values.shape[0], 576, self.config.hidden_size)


class KVCache(object):
    def __init__(self) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

    def num_items(self) -> int:
        if len(self.key_cache) == 0:
            return 0
        else:
            return self.key_cache[0].shape[-2]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass


class GemmaConfig:
    def __init__(
        self,
        vocab_size,
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        head_dim=256,
        max_position_embeddings=8192,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        pad_token_id=None,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id


class PaliGemmaConfig:
    def __init__(
        self,
        vision_config=None,
        text_config=None,
        ignore_index=-100,
        image_token_index=256000,
        vocab_size=257152,
        projection_dim=2048,
        hidden_size=2048,
        pad_token_id=None,
        **kwargs,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.is_encoder_decoder = False
        self.pad_token_id = pad_token_id

        self.vision_config = SiglipVisionConfig(**vision_config)
        self.text_config = GemmaConfig(**text_config, pad_token_id=pad_token_id)
        self.vocab_size = self.text_config.vocab_size

        self.text_config.num_image_tokens = (
            self.vision_config.image_size // self.vision_config.patch_size
        ) ** 2
        self.vision_config.projection_dim = projection_dim

class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        return (output * self.weight).type_as(x)

class GemmaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=False
        )
    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        gate_output = F.gelu(self.gate_proj(hidden_states))
        up_output = self.up_proj(hidden_states)
        return self.down_proj(gate_output * up_output)


class GemmaAttention(nn.Module):
    def __init__(
            self,
            config: GemmaConfig,
            layer_idx: int,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.rotary_emb = None
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return hidden_states

class GemmaDecoderLayer(nn.Module):
    def __init__(
            self,
            config: GemmaConfig,
            layer_idx: int,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = GemmaAttention(
            config=config,
            layer_idx=layer_idx,
        )
        self.mlp = GemmaMLP(config)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
       return hidden_states

class GemmaModel(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([GemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self):
        return self.embed_tokens
    
    def forward(
        self,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> torch.FloatTensor:
        hidden_states = inputs_embeds
        if hidden_states is None:
             raise ValueError("You have to specify inputs_embeds")
        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype, device=hidden_states.device)
        hidden_states = hidden_states * normalizer
        
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(hidden_states, **kwargs)
        
        hidden_states = self.norm(hidden_states)
        return hidden_states

class GemmaForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = GemmaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    
    def get_input_embeddings(self):
        return self.model.embed_tokens
    
    def tie_weights(self):
        self.lm_head.weight = self.model.embed_tokens.weight
    
    def forward(
            self,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            **kwargs,
    ) -> dict:
        hidden_states = self.model(inputs_embeds=inputs_embeds, **kwargs)
        logits = self.lm_head(hidden_states)
        return {"logits": logits}

class PaliGemmaMultiModalProjector(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.linear = nn.Linear(config.vision_config.hidden_size, config.vision_config.projection_dim, bias=True)

    def forward(self, image_features):
        hidden_state = self.linear(image_features)
        return hidden_state
    
class PaliGemmaForConditionalGeneration(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config = config
        self.vision_tower = SiglipVisionModel(config.vision_config)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)
        self.vocab_size = config.vocab_size
        self.language_model = GemmaForCausalLM(config.text_config)
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

    def tie_weights(self):
        return self.language_model.tie_weights()

    def _merge_input_ids_with_image_features(
        self,
        image_features: torch.FloatTensor,
        inputs_embeds: torch.FloatTensor,
        input_ids: torch.LongTensor,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, torch.Tensor, torch.LongTensor]:
        final_embedding = torch.zeros_like(inputs_embeds)
        return final_embedding, torch.ones_like(input_ids), torch.arange(input_ids.shape[1], device=input_ids.device)


    def forward(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        selected_image_feature = self.vision_tower(pixel_values.to(inputs_embeds.dtype))
        image_features = self.multi_modal_projector(selected_image_feature)
        
        inputs_embeds, attention_mask, position_ids = self._merge_input_ids_with_image_features(
            image_features, inputs_embeds, input_ids, attention_mask=attention_mask
        )
        
        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        return outputs

dummy_vision_config_dict = {
    "hidden_size": 1024, "image_size": 224, "intermediate_size": 4096,
    "num_attention_heads": 16, "num_hidden_layers": 2, "patch_size": 14,
}

dummy_text_config_dict = {
    "vocab_size": 257152, "hidden_size": 2048, "intermediate_size": 8192,
    "num_hidden_layers": 2, "num_attention_heads": 8, "num_key_value_heads": 1,
    "head_dim": 256, "max_position_embeddings": 4096,
}

paligemma_config = PaliGemmaConfig(
    vision_config=dummy_vision_config_dict,
    text_config=dummy_text_config_dict,
    hidden_size=2048,
    pad_token_id=0,
)


print("--- GemmaForCausalLM Architecture ---")
gemma_model = GemmaForCausalLM(config=paligemma_config.text_config)
print(gemma_model)
print("-" * 80)

print("\n--- PaliGemmaForConditionalGeneration Architecture ---")
pali_model = PaliGemmaForConditionalGeneration(config=paligemma_config)
print(pali_model)
print("-" * 80)
