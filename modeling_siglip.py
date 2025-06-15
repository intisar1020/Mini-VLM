# torch stuff
import torch
import torch.nn as nn

# others
from typing import Optional, Tuple


##################################################
class SiglipVisionConfig:
    """
    Configuration class for the Siglip Vision Transformer model.

    Args:
        hidden_size (int): Dimensionality of the encoder layers and the pooler layer.
                           Also known as `embed_dim`.
        intermediate_size (int): Dimensionality of the "intermediate" (feed-forward) layer in the MLP.
        num_hidden_layers (int): Number of hidden layers in the Transformer encoder.
        num_attention_head (int): Number of attention heads for each attention layer in the Transformer encoder.
        num_channels (int): Number of input channels for the image (e.g., 3 for RGB images).
        image_size (int): The size (resolution) of the input images.
        patch_size (int): The size (resolution) of the patches to be extracted from the input images.
        layer_norm_eps (float): The epsilon used by the layer normalization layers.
        attention_dropout (float): The dropout ratio for the attention probabilities.
        num_image_tokens (int, optional): The number of image tokens. If None, it will be derived.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        num_hidden_layers: int = 12,
        num_attention_head: int = 12,
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
        self.num_attention_heads = num_attention_head
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.num_image_tokens = num_image_tokens


###########################################################
class SiglipVisionEmbeddings(nn.Module):
    """
    Computes patch embeddings from input pixel values and adds positional embeddings.
    """

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        # Convolutional layer to extract patch embeddings
        # Input: (batch_size, num_channels, image_size, image_size)
        # Output: (batch_size, embed_dim, num_patches_h, num_patches_w)
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=config.patch_size,
            padding="valid",
        )

        # Calculate the number of patches
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches  # Each patch gets a position

        # Positional embedding layer
        # Input: (1, num_positions) for position_ids
        # Output: (1, num_positions, embed_dim)
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)

        # Register position_ids as a buffer so it's part of the model's state
        # but not a learnable parameter.
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )
        # Note: Siglip does not use a [CLS] token embedding.

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        """
        Forward pass for the SiglipVisionEmbeddings.

        Args:
            pixel_values (torch.FloatTensor): Input image tensor.
                                              Shape: (batch_size, num_channels, image_height, image_width)
                                              Example: (B, 3, 224, 224)

        Returns:
            torch.Tensor: Patch embeddings combined with positional embeddings.
                          Shape: (batch_size, num_patches, embed_dim)
                          Example: (B, 196, 768) for 224x224 image, 16x16 patches, 768 hidden_size.
        """
        # Get dimensions of the input image
        # pixel_values shape: (batch_size, num_channels, height, width)
        batch_size, num_channels, height, width = pixel_values.shape

        # Apply convolutional patch embedding
        # patch_embeds shape: (batch_size, embed_dim, num_patches_height, num_patches_width)
        # Example: (B, 768, 14, 14) from (B, 3, 224, 224) with patch_size=16
        patch_embeds = self.patch_embedding(pixel_values)

        # Flatten the spatial dimensions of the patch embeddings
        # from (B, embed_dim, num_patches_h, num_patches_w) to (B, embed_dim, num_patches)
        # Example: (B, 768, 196)
        embeddings = patch_embeds.flatten(2)

        # Transpose to get the sequence dimension in the middle
        # from (B, embed_dim, num_patches) to (B, num_patches, embed_dim)
        # Example: (B, 196, 768)
        embeddings = embeddings.transpose(1, 2)

        # Add positional embeddings to the patch embeddings
        # self.position_ids shape: (1, num_patches)
        # self.position_embedding(self.position_ids) shape: (1, num_patches, embed_dim)
        # embeddings shape: (batch_size, num_patches, embed_dim)
        embeddings = embeddings + self.position_embedding(self.position_ids)

        return embeddings


class SiglipMLP(nn.Module):
    """
    A simple two-layer Multi-Layer Perceptron (MLP) with GELU activation.
    Used within each Transformer encoder layer.
    """

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        # First linear layer
        # Input: (..., hidden_size)
        # Output: (..., intermediate_size)
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        # Second linear layer
        # Input: (..., intermediate_size)
        # Output: (..., hidden_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        # Siglip paper uses GELU activation, which is applied implicitly in the original implementation
        # For full compatibility, an activation function (e.g., nn.GELU()) should be added between fc1 and fc2.
        # However, for simplicity and matching the provided structure, it's omitted here.

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the SiglipMLP.

        Args:
            hidden_states (torch.Tensor): Input tensor.
                                          Shape: (batch_size, sequence_length, hidden_size)
                                          Example: (B, 196, 768)

        Returns:
            torch.Tensor: Output tensor after MLP operations.
                          Shape: (batch_size, sequence_length, hidden_size)
                          Example: (B, 196, 768)
        """
        # Apply first linear layer
        # hidden_states shape: (B, S, intermediate_size)
        hidden_states = self.fc1(hidden_states)
        # Apply second linear layer
        # hidden_states shape: (B, S, hidden_size)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class SiglipAttention(nn.Module):
    """
    Multi-head Self-Attention mechanism for the Siglip Vision Transformer.
    """

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = self.config.hidden_size
        self.num_heads = config.num_attention_heads
        # Dimension of each attention head
        self.head_dim = self.embed_dim // self.num_heads
        # Scaling factor for dot product attention
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        # Linear layers for Query, Key, and Value projections
        # Input: (..., embed_dim)
        # Output: (..., embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)  # Key projection
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)  # Value projection
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)  # Query projection
        # Output linear layer after concatenating attention heads
        # Input: (..., embed_dim)
        # Output: (..., embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for the SiglipAttention.

        Args:
            hidden_states (torch.Tensor): Input tensor from the previous layer.
                                          Shape: (batch_size, sequence_length, embed_dim)
                                          Example: (B, 196, 768)

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - attn_output (torch.Tensor): Output tensor after attention.
                                              Shape: (batch_size, sequence_length, embed_dim)
                                              Example: (B, 196, 768)
                - attn_weights (torch.Tensor): Attention weights (optional).
                                               Shape: (batch_size, num_heads, sequence_length, sequence_length)
                                               Example: (B, 12, 196, 196)
        """
        batch_size, seq_len, _ = hidden_states.size()

        # Project hidden states to query, key, and value
        # query_states, key_states, value_states all have shape: (batch_size, seq_len, embed_dim)
        # Example: (B, 196, 768)
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape and transpose for multi-head attention
        # The view operation splits the last dimension (embed_dim) into (num_heads, head_dim)
        # query_states/key_states/value_states after view: (batch_size, seq_len, num_heads, head_dim)
        # Example: (B, 196, 12, 64) for (B, 196, 768) with 12 heads.
        query_states = query_states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        key_states = key_states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        value_states = value_states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        )

        # Transpose to bring num_heads dimension to the second position
        # This makes it easier for batch matrix multiplication across heads
        # query_states/key_states/value_states after transpose: (batch_size, num_heads, seq_len, head_dim)
        # Example: (B, 12, 196, 64)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # Calculate attention scores (query @ key.T) and scale
        # query_states: (B, num_heads, seq_len, head_dim)
        # key_states.transpose(2,3): (B, num_heads, head_dim, seq_len)
        # attn_weights shape: (batch_size, num_heads, seq_len, seq_len)
        # Example: (B, 12, 196, 196)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale

        if attn_weights.size() != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError("Attention weight size mismatch")

        # Apply softmax to get attention probabilities
        # attn_weights shape remains: (batch_size, num_heads, seq_len, seq_len)
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)

        # Multiply attention probabilities with value states
        # attn_weights: (B, num_heads, seq_len, seq_len)
        # value_states: (B, num_heads, seq_len, head_dim)
        # attn_output after matmul: (batch_size, num_heads, seq_len, head_dim)
        # Example: (B, 12, 196, 64)
        attn_output = torch.matmul(attn_weights, value_states)

        # Transpose back to combine heads
        # from (batch_size, num_heads, seq_len, head_dim) to (batch_size, seq_len, num_heads, head_dim)
        # Example: (B, 196, 12, 64)
        attn_output = attn_output.transpose(1, 2).contiguous()

        # Reshape to concatenate the heads into a single embedding dimension
        # from (batch_size, seq_len, num_heads, head_dim) to (batch_size, seq_len, num_heads * head_dim)
        # num_heads * head_dim is equal to embed_dim
        # Example: (B, 196, 768)
        attn_output = attn_output.reshape(batch_size, seq_len, self.num_heads * self.head_dim)

        # Apply final linear projection
        # attn_output shape remains: (batch_size, seq_len, embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class SiglipEncoderLayer(nn.Module):
    """
    A single encoder layer of the Siglip Vision Transformer.
    Consists of self-attention and a feed-forward network, each with residual connections and layer normalization.
    """

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        # Layer normalization before attention
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        # Layer normalization before MLP
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a single SiglipEncoderLayer.

        Args:
            hidden_states (torch.Tensor): Input tensor from the previous layer or embeddings.
                                          Shape: (batch_size, sequence_length, embed_dim)
                                          Example: (B, 196, 768)

        Returns:
            torch.Tensor: Output tensor after passing through the encoder layer.
                          Shape: (batch_size, sequence_length, embed_dim)
                          Example: (B, 196, 768)
        """
        # Self-Attention Block with Pre-LayerNorm and Residual Connection
        residual = hidden_states  # Store for residual connection
        # Normalize before attention
        # hidden_states shape: (B, S, E)
        hidden_states = self.layer_norm1(hidden_states)
        # Apply self-attention
        # hidden_states shape: (B, S, E)
        hidden_states, _ = self.self_attn(hidden_states=hidden_states)
        # Add residual connection
        # hidden_states shape: (B, S, E)
        hidden_states += residual

        # MLP Block with Pre-LayerNorm and Residual Connection
        residual = hidden_states  # Store for residual connection
        # Normalize before MLP
        # hidden_states shape: (B, S, E)
        hidden_states = self.layer_norm2(hidden_states)
        # Apply MLP
        # hidden_states shape: (B, S, E)
        hidden_states = self.mlp(hidden_states)
        # Add residual connection
        # hidden_states shape: (B, S, E)
        hidden_states = residual + hidden_states
        return hidden_states


class SiglipEncoder(nn.Module):
    """
    The full encoder stack of the Siglip Vision Transformer.
    Composed of multiple `SiglipEncoderLayer` instances.
    """

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        # Create a list of encoder layers
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the SiglipEncoder.

        Args:
            inputs_embeds (torch.Tensor): Input embeddings from the embedding layer.
                                          Shape: (batch_size, num_patches, embed_dim)
                                          Example: (B, 196, 768)

        Returns:
            torch.Tensor: The hidden states after passing through all encoder layers.
                          Shape: (batch_size, num_patches, embed_dim)
                          Example: (B, 196, 768)
        """
        hidden_states = inputs_embeds

        for encoder_layer in self.layers:
            # Each encoder layer takes (B, S, E) and outputs (B, S, E)
            hidden_states = encoder_layer(hidden_states)

        return hidden_states


class SiglipVisionTransformer(nn.Module):
    """
    The core Vision Transformer model for Siglip.
    Combines patch embeddings, encoder layers, and a final layer normalization.
    """

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        # Patch and positional embeddings
        self.embeddings = SiglipVisionEmbeddings(config)
        # Encoder stack
        self.encoder = SiglipEncoder(config)
        # Final layer normalization after the encoder
        self.post_layernorm = nn.LayerNorm(embed_dim)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the SiglipVisionTransformer.

        Args:
            pixel_values (torch.Tensor): Input image tensor.
                                         Shape: (batch_size, num_channels, image_height, image_width)
                                         Example: (B, 3, 224, 224)

        Returns:
            torch.Tensor: The final hidden states of the transformer.
                          Shape: (batch_size, num_patches, embed_dim)
                          Example: (B, 196, 768)
        """
        # Compute patch and positional embeddings
        # hidden_states shape: (batch_size, num_patches, embed_dim)
        hidden_states = self.embeddings(pixel_values)
        # Pass through the encoder stack
        # last_hidden_state shape: (batch_size, num_patches, embed_dim)
        last_hidden_state = self.encoder(inputs_embeds=hidden_states)
        # Apply final layer normalization
        # last_hidden_state shape remains: (batch_size, num_patches, embed_dim)
        last_hidden_state = self.post_layernorm(last_hidden_state)
        return last_hidden_state


class SiglipVisionModel(nn.Module):
    """
    The full Siglip Vision Model, encapsulating the SiglipVisionTransformer.
    This module primarily serves as an entry point for the vision component.
    """

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the SiglipVisionModel.

        Args:
            pixel_values (torch.Tensor): Input image tensor.
                                         Shape: (batch_size, num_channels, image_height, image_width)
                                         Example: (B, 3, 224, 224)

        Returns:
            torch.Tensor: The output of the vision transformer.
                          Shape: (batch_size, num_patches, embed_dim)
                          Example: (B, 196, 768)
        """
        return self.vision_model(pixel_values)


if __name__ == "__main__":
    # --- Testing the model components ---
    print("--- Siglip Vision Model Testing ---")

    # Initialize configuration
    config = SiglipVisionConfig(
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=1,  # Using 1 layer for quick testing
        num_attention_head=12,
        num_channels=3,
        image_size=224,
        patch_size=16,
    )
    print("\n--- Configuration Parameters ---")
    for k, v in config.__dict__.items():
        if not k.startswith("_"):  # Exclude private attributes
            print(f"{k}: {v}")
    print("-" * 30)

    # Instantiate the full Siglip Vision Model
    model = SiglipVisionModel(config)
    print("\n--- Model Architecture ---")
    print(model)
    print("-" * 30)

    # Create a dummy input tensor for testing
    # Batch size = 2, 3 channels (RGB), 224x224 image size
    input_data = torch.rand((2, 3, 224, 224))
    print("\n--- Input Tensor Shape ---")
    print(f"Input: {input_data.shape}")
    print("-" * 30)

    # Perform a forward pass
    print("\n--- Running Forward Pass ---")
    with torch.no_grad():  # Disable gradient calculation for inference
        output = model(input_data)

    print("\n--- Output Tensor Shape ---")
    print(f"Output shape: {output.shape}")  # Expected: (2, 196, 768)
    print("-" * 30)

    # Verify expected output shape based on configuration
    expected_num_patches = (config.image_size // config.patch_size) ** 2
    expected_output_shape = (input_data.shape[0], expected_num_patches, config.hidden_size)

    if output.shape == expected_output_shape:
        print("Output shape matches expected shape: SUCCESS!")
    else:
        print(f"Output shape mismatch! Expected {expected_output_shape}, Got {output.shape}")

    print("\n--- End of Testing ---")
