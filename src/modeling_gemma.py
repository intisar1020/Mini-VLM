import torch
from torch import nn
from typing import Optional, Tuple, List
from modeling_siglip import SiglipVisionConfig, SiglipVisionModel


class KVCache(object):
    def __init__(self) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

    def num_items(self) -> int:
        if len(self.key_cache) == 0:
            return 0
        else:
            # returns total number of tokens in the cache
            # [batch_size, num_heads, seq_len, head_dim]
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
        self.max_postion_embeddings = max_position_embeddings
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
        self.vision_config = vision_config
        self.is_encoder_decoder = False
        self.pad_token_id = pad_token_id

        self.vision_config = SiglipVisionConfig(**vision_config)
        self.text_config = text_config

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
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

class GemmaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
        )
        self.up_proj = nn.Linear(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
        )
        self.down_proj = nn.Linear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
        )
    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        # hidden_states: [Batch_Size, Seq_Len, Hidden_Size]
        gate_output = torch.gelu(self.gate_proj(hidden_states), approximate="tanh")
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

        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=False,
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
        )
        self.rotary_emb = None
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        **kwargs,
    ) -> torch.Tensor:
        pass

class GemmaDecoderLayer:
    def __init__(
            self,
            config: GemmaConfig,
            layer_idx: int,
    ):
        self.hidden_size = config.hidden_size
        self.self_attn = GemmaAttention

class GemmaModel(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, 
            config.hidden_size, 
            self.padding_idx
        )
        self.layers = nn.ModuleList(
            [GemmaDecoderLayer(config, layer_idx)
             for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self):
        return self.embed_tokens
    
    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.FloatTensor:
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = inputs_embeds
        # [Batch_Size, Seq_Len, Hidden_Size]
        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype)
        hidden_states = hidden_states * normalizer


class GemmaForCausalLM:
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = GemmaModel(config)
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )
    
    def get_input_embeddings(self):
        return self.model.embed_tokens
    
    def tie_weights(self):
        self.lm_head.weight = self.model.embed_tokens.weight
    
    def forward(
            self,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            kv_cache: Optional[KVCache] = None,
    ) -> Tuple:
        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
        )
        hidden_states = outputs
        logits = self.lm_head(hidden_states)
        return_data = {
            "logits": logits
        }
        if kv_cache is not None:
            return_data["kv_cache"] = kv_cache
        return return_data

class PaliGemmaMultiModalProjector(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.Linear = nn.Linear(
            config.vision_config.hidden_size,
            config.vision_config.projection_dim,
            bias=True,
        )

    def forward(self, image_features):
        hidden_state = self.Linear(image_features)
        return hidden_state
    
class PaliGemmaForConditionalGeneration(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config = config
        self.vision_tower = SiglipVisionModel(config.vision_config)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)
        self.vocab_sie = config.vocab_size

        language_model = GemmaForCausalLM(config.text_config)
        self.language_model = language_model

        if self.config.pad_token_id:
            self.pad_token_id = self.config.pad_token_id
        else:
            self.pad_token_id = -1

    def tie_weights(self):
        return self.language_model.tie_weights()

    def _merge_input_ids_with_image_features(
        self,
        image_features: torch.FloatTensor = None,
        inputs_embeds: torch.FloatTensor = None,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.FloatTensor, torch.Tensor, torch.LongTensor]:
        """Merges the image features with the input embeddings and returns the final embeddings, attention mask, and position ids.

        Args:
            image_features (torch.FloatTensor, optional): _description_. Defaults to None.
            inputs_embeds (torch.FloatTensor, optional): _description_. Defaults to None.
            input_ids (torch.LongTensor, optional): _description_. Defaults to None.
            attention_mask (Optional[torch.Tensor], optional): _description_. Defaults to None.
            kv_cache (Optional[KVCache], optional): _description_. Defaults to None.

        Returns:
            Tuple[torch.FloatTensor, torch.Tensor, torch.LongTensor]: _description_
        """
        _, _, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        dtype, device = inputs_embeds.dtype, inputs_embeds.device
        scaled_image_features = image_features / (self.config.hidden_size**0.5)

        # now we will combine the embedding of the image tokens and the text token
        # each sequence will have the embedding for image (that is extracted by the vision tower)
        # and the embedding for the text token that is extracted the embedding extractor of the language model.

        # shape of inputs_embeds: (batch_size, sequence_length, embed_dim)
        # shape of final_embedding: (batch_size, sequence_length, embed_dim)
        final_embedding = torch.zeroes(
            batch_size,
            sequence_length,
            embed_dim,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )
        image_mask = input_ids == self.config.image_token_index
        pad_mask = input_ids == self.pad_token_id
        text_mask = (input_ids != self.config.image_token_index) & (
            input_ids != self.pad_token_id
        )

        text_mask_expanded = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        pad_mask_expanded = pad_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        image_mask_expanded = image_mask.unsqueeze(-1).expand(-1, -1, embed_dim)

        # fill the final embedding with the image features
        final_embedding = torch.where(
            text_mask_expanded, inputs_embeds, final_embedding
        )
        final_embedding = final_embedding.masked_scatter(
            image_mask_expanded,
            scaled_image_features,
        )
        final_embedding = torch.where(
            pad_mask_expanded,
            torch.zeros_like(final_embedding, dtype=dtype, device=device),
            final_embedding,
        )

        dtype, device = inputs_embeds.dtype, inputs_embeds.device
        # min_dtype = torch.finfo(dtype).min
        # shape of inputs_embeds: (batch_size, sequence_length, embed_dim)
        q_len = inputs_embeds.shape[1]  # sequence length

        if kv_cache is None or kv_cache.items() == 0:
            causal_mask = torch.full(
                (batch_size, q_len, q_len),
                fill_value=0,
                dtype=dtype,
                device=device,
            )
        else:
            assert q_len == 1
            kv_len = kv_cache.num_items() + 1
            # kv_cache.num_items() returns the number of tokens in the cache
            # we add 1 to account for the current token
            # shape of causal_mask: (batch_size, 1, kv_len)
            causal_mask = torch.full(
                (batch_size, q_len, kv_len),
                fill_value=0,
                dtype=dtype,
                device=device,
            )
        causal_mask = causal_mask.unsqueeze(1)  # [batch_size, 1, q_len, kv_len]

        if kv_cache is not None and kv_cache.num_items() > 0:
            position_ids = attention_mask.cumsum(-1)[:, -1]
            # dimension of attention_mask: (batch_size, seq_len)
            # here cumsum(-1) gives us the cumulative sum along the last dimension
            # [:, -1] gives us the last element of the cumulative sum for each batch
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0) 
        else:
            position_ids = (
                (attention_mask.cumsum(-1))
                .masked_fill_((attention_mask == 0), 1)
                .to(device)
            )

        return final_embedding, causal_mask, position_ids

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple:
        assert torch.all(attention_mask == 1), "We do not pad the input"

        # 1. extract the input embeddings.
        # shape: (batch_size, seq_len, hidden_size)
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        # 2. Merge text and images.
        # shape: (batch_size),channels, height, width) -> (batch_size, num_patches, embed_dim)
        selected_image_feature = self.vision_tower(pixel_values.to(inputs_embeds.dtype))
        # 3. resize the image feature into size compatible with the LLM
        image_features = self.multi_modal_projector(selected_image_feature)

        # 4. merge the token from vision model to the text token (fill up place-holder)
        inputs_embeds, attention_mask, position_ids = (
            self._merge_input_ids_with_image_features(
                image_features,  # from vit.
                inputs_embeds,  # from llm
                input_ids,  # from tokenizer.
                attention_mask,  # from tokenizer.
                kv_cache,  # cache for optimality.
            )
        )
        # To do: Implement the language model.
        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
        )

        return outputs


config = SiglipVisionConfig()
model = SiglipVisionModel(config=config)
print(model)

pali = PaliGemmaForConditionalGeneration()
gemma_model = GemmaForCausalLM(config=pali.config.text_config)
print(gemma_model)
