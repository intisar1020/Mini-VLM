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
        hidden_size (int): Dimensionality of the encoder layers and the pooler layer (embed_dim).
        intermediate_size (int): Dimensionality of the feed-forward layer in the MLP.
        num_hidden_layers (int): Number of encoder layers.
        num_attention_head (int): Number of attention heads per layer.
        num_channels (int): Number of image input channels (usually 3 for RGB).
        image_size (int): Height and width of the input image (assumes square).
        patch_size (int): Size of the image patches.
        layer_norm_eps (float): Epsilon for layer normalization.
        attention_dropout (float): Dropout probability in attention.
        num_image_tokens (Optional[int]): Number of image tokens (usually computed).
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
    ) -> None:
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


##################################################
class SiglipVisionEmbeddings(nn.Module):
    """
    Computes patch embeddings from input images and adds positional encodings.
    """

    def __init__(self, config: SiglipVisionConfig) -> None:
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
            padding="valid",
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.position_embedding = nn.Embedding(self.num_patches, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_patches).expand((1, -1)), persistent=False)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        embeddings = self.patch_embedding(pixel_values).flatten(2).transpose(1, 2)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


##################################################
class SiglipMLP(nn.Module):
    """
    Feed-forward network used in each Transformer layer.
    """

    def __init__(self, config: SiglipVisionConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.activation(self.fc1(x)))


##################################################
class SiglipAttention(nn.Module):
    """
    Multi-head self-attention module.
    """

    def __init__(self, config: SiglipVisionConfig) -> None:
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.dropout = nn.Dropout(config.attention_dropout)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, S, E = hidden_states.size()

        q = self.q_proj(hidden_states).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).reshape(B, S, E)
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights


##################################################
class SiglipEncoderLayer(nn.Module):
    """
    A single Transformer encoder block with attention and MLP.
    """

    def __init__(self, config: SiglipVisionConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = SiglipAttention(config)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x)
        x = x + residual

        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + residual
        return x


##################################################
class SiglipEncoder(nn.Module):
    """
    Stack of Transformer encoder layers.
    """

    def __init__(self, config: SiglipVisionConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList([SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


##################################################
class SiglipVisionTransformer(nn.Module):
    """
    Vision Transformer model combining embeddings and encoder.
    """

    def __init__(self, config: SiglipVisionConfig) -> None:
        super().__init__()
        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = self.embeddings(pixel_values)
        x = self.encoder(x)
        x = self.norm(x)
        return x


##################################################
class SiglipVisionModel(nn.Module):
    """
    Wrapper for the Siglip vision transformer.
    """

    def __init__(self, config: SiglipVisionConfig) -> None:
        super().__init__()
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.vision_model(pixel_values)


##################################################
if __name__ == "__main__":
    print("\n--- Siglip Vision Model Testing ---")

    # Configuration
    config = SiglipVisionConfig(
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=1,  # For quick test
        num_attention_head=12,
        num_channels=3,
        image_size=224,
        patch_size=16,
    )

    print("\n--- Configuration ---")
    for k, v in config.__dict__.items():
        if not k.startswith("_"):
            print(f"{k}: {v}")

    print("\n--- Initializing Model ---")
    model = SiglipVisionModel(config)
    print(model)

    # Dummy input
    input_tensor = torch.rand((2, 3, 224, 224))  # batch=2
    print("\n--- Input Shape ---")
    print(input_tensor.shape)

    # Forward pass
    print("\n--- Running Forward Pass ---")
    with torch.no_grad():
        output = model(input_tensor)

    print("\n--- Output Shape ---")
    print(output.shape)

    expected_patches = (config.image_size // config.patch_size) ** 2
    expected_shape = (input_tensor.shape[0], expected_patches, config.hidden_size)
    assert output.shape == expected_shape, f"Shape mismatch: expected {expected_shape}, got {output.shape}"

    print("\n--- SUCCESS: Output shape is correct ---")
