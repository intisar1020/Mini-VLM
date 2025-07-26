from typing import Optional, Tuple
import torch
import torch.nn as nn

class SiglipVisionConfig:
    """
    Configuration class for the Siglip Vision Transformer model.

    Args:
        hidden_size (int): Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (int): Dimensionality of the "intermediate" (feed-forward) layer in the MLP.
        num_hidden_layers (int): Number of hidden layers in the Transformer encoder.
        num_attention_heads (int): Number of attention heads for each attention layer.
        num_channels (int): Number of input image channels.
        image_size (int): Resolution of the input image.
        patch_size (int): Size of patches to be extracted from the input image.
        layer_norm_eps (float): The epsilon used by the layer normalization layers.
        attention_dropout (float): The dropout ratio for the attention probabilities.
        num_image_tokens (Optional[int]): The number of image tokens. If None, it is calculated from image_size and patch_size.
    """
    def __init__(
        self,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        num_channels: int = 3,
        image_size: int = 224,
        patch_size: int = 16,
        layer_norm_eps: float = 1e-6,
        attention_dropout: float = 0.0,
        num_image_tokens: Optional[int] = None,
        **kwargs
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.num_image_tokens = num_image_tokens


class SiglipVisionEmbeddings(nn.Module):
    """
    Converts input image pixel values into patch embeddings and adds positional embeddings.
    """
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
            stride=self.patch_size,
            padding="valid",
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        """
        Args:
            pixel_values (torch.FloatTensor): Input pixel values of shape `(batch_size, num_channels, height, width)`.

        Returns:
            torch.Tensor: Patch embeddings with positional embeddings added, of shape
                          `(batch_size, num_patches, embed_dim)`.
        """
        _, _, height, width = pixel_values.shape
        patch_embeds = self.patch_embedding(pixel_values)
        embeddings = patch_embeds.flatten(2)
        embeddings = embeddings.transpose(1, 2)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


class SiglipAttention(nn.Module):
    """
    Multi-headed attention mechanism for the Siglip Vision Transformer.
    """
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            hidden_states (torch.Tensor): Input hidden states of shape `(batch_size, sequence_length, embed_dim)`.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: A tuple containing:
                - attn_output (torch.Tensor): The attention output of shape `(batch_size, sequence_length, embed_dim)`.
                - attn_weights (Optional[torch.Tensor]): The attention weights of shape
                                                         `(batch_size, num_heads, sequence_length, sequence_length)`.
        """
        batch_size, seq_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = (torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale)

        if attn_weights.size() != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, seq_len, seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, seq_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class SiglipMLP(nn.Module):
    """
    Multi-layer Perceptron (MLP) block used in the Transformer encoder layers.
    """
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (torch.Tensor): Input hidden states of shape `(batch_size, sequence_length, embed_dim)`.

        Returns:
            torch.Tensor: Output of the MLP, with shape `(batch_size, sequence_length, embed_dim)`.
        """
        hidden_states = self.fc1(hidden_states)
        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh")
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class SiglipEncoderLayer(nn.Module):
    """
    A single encoder layer of the Siglip Vision Transformer.
    """
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (torch.Tensor): Input hidden states of shape `(batch_size, sequence_length, embed_dim)`.

        Returns:
            torch.Tensor: Output of the encoder layer, with shape `(batch_size, sequence_length, embed_dim)`.
        """
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states=hidden_states)
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class SiglipEncoder(nn.Module):
    """
    The encoder stack of Siglip Vision Transformer, consisting of multiple EncoderLayers.
    """
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            inputs_embeds (torch.Tensor): Input embeddings of shape `(batch_size, sequence_length, embed_dim)`.

        Returns:
            torch.Tensor: The output hidden states from the encoder, with shape
                          `(batch_size, sequence_length, embed_dim)`.
        """
        hidden_states = inputs_embeds

        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states)

        return hidden_states


class SiglipVisionTransformer(nn.Module):
    """
    The main Vision Transformer model for Siglip, integrating embeddings and the encoder.
    """
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values (torch.Tensor): Input pixel values of shape `(batch_size, num_channels, height, width)`.

        Returns:
            torch.Tensor: The final hidden states from the vision transformer, of shape
                          `(batch_size, num_patches, embed_dim)`.
        """
        hidden_states = self.embeddings(pixel_values)
        last_hidden_state = self.encoder(inputs_embeds=hidden_states)
        last_hidden_state = self.post_layernorm(last_hidden_state)
        return last_hidden_state


class SiglipVisionModel(nn.Module):
    """
    Siglip Vision Model wrapper.
    """
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values (torch.Tensor): Input pixel values of shape `(batch_size, num_channels, height, width)`.

        Returns:
            torch.Tensor: The output of the vision model, which is the last hidden state from the transformer,
                          of shape `(batch_size, num_patches, embed_dim)`.
        """
        return self.vision_model(pixel_values=pixel_values)

if __name__ == "__main__":
    print("\n--- Siglip Vision Model Test ---")

    # Define a minimal configuration for testing
    config = SiglipVisionConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_channels=3,
        image_size=32,
        patch_size=8,
        layer_norm_eps=1e-6,
        attention_dropout=0.0
    )

    print("\n--- Configuration ---")
    for k, v in config.__dict__.items():
        if not k.startswith("_"):
            print(f"{k}: {v}")

    print("\n--- Initializing Model ---")
    model = SiglipVisionModel(config)
    print(model)

    # Test input
    batch_size = 2
    test_input = torch.randn(batch_size, config.num_channels, config.image_size, config.image_size)
    print(f"\n--- Test Input Shape: {test_input.shape} ---")

    # Perform forward pass
    print("\n--- Running Forward Pass ---")
    with torch.no_grad():
        output = model(test_input)

    print(f"--- Test Output Shape: {output.shape} ---")

    # Assert expected output shape
    expected_num_patches = (config.image_size // config.patch_size) ** 2
    expected_output_shape = (batch_size, expected_num_patches, config.hidden_size)

    print(f"--- Expected Output Shape: {expected_output_shape} ---")

    assert output.shape == expected_output_shape, \
        f"Output shape mismatch! Expected {expected_output_shape}, but got {output.shape}"

    print("\n--- Input-Output Test Passed Successfully! ---")
    print(f"Model output data type: {output.dtype}")
    print(f"Model output device: {output.device}")

    # Optional: Test with a different batch size
    print("\n--- Testing with a different batch size ---")
    test_input_large_batch = torch.randn(4, config.num_channels, config.image_size, config.image_size)
    with torch.no_grad():
        output_large_batch = model(test_input_large_batch)
    
    expected_output_shape_large_batch = (4, expected_num_patches, config.hidden_size)
    print(f"--- Test Input Shape (large batch): {test_input_large_batch.shape} ---")
    print(f"--- Test Output Shape (large batch): {output_large_batch.shape} ---")
    assert output_large_batch.shape == expected_output_shape_large_batch, \
        f"Output shape mismatch for large batch! Expected {expected_output_shape_large_batch}, but got {output_large_batch.shape}"
    print("\n--- Large Batch Test Passed Successfully! ---")