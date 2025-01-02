"""Tests for multimodal processing components."""

import torch
import pytest
from nncore.multimodal import (
    SpeechEncoder,
    VisionProcessor,
    TextProcessor,
    CrossModalFusion,
)


def test_speech_encoder():
    """Test SpeechEncoder functionality."""
    batch_size = 16
    seq_length = 100
    input_dim = 80
    hidden_dim = 256

    encoder = SpeechEncoder(input_dim=input_dim, hidden_dim=hidden_dim)

    # Test basic forward pass
    x = torch.randn(batch_size, seq_length, input_dim)
    output = encoder(x)

    expected_dim = hidden_dim * len(
        encoder.conv_layers[0]
    )  # hidden_dim * num_kernel_sizes
    assert output.shape == (batch_size, seq_length, expected_dim)

    # Test with mask
    mask = torch.zeros(batch_size, seq_length, dtype=torch.bool)
    mask[:, seq_length // 2 :] = True  # Mask second half
    output_masked = encoder(x, mask)

    assert output_masked.shape == (batch_size, seq_length, expected_dim)
    assert not torch.equal(output, output_masked)


def test_vision_processor():
    """Test VisionProcessor functionality."""
    batch_size = 8
    channels = 3
    height = 224
    width = 224

    processor = VisionProcessor(in_channels=channels, base_channels=64, num_scales=3)

    # Test forward pass
    x = torch.randn(batch_size, channels, height, width)
    features = processor(x)

    # Check number of feature scales
    assert len(features) == processor.num_scales

    # Check feature dimensions
    expected_channels = [64 * 2**i for i in range(1, processor.num_scales + 1)]
    expected_sizes = [
        (height // 2 ** (i + 2), width // 2 ** (i + 2))
        for i in range(processor.num_scales)
    ]

    for feat, channels, (h, w) in zip(features, expected_channels, expected_sizes):
        assert feat.shape == (batch_size, channels, h, w)


def test_text_processor():
    """Test TextProcessor functionality."""
    batch_size = 32
    seq_length = 50
    vocab_size = 30000
    embed_dim = 512

    processor = TextProcessor(vocab_size=vocab_size, embed_dim=embed_dim)

    # Test basic forward pass
    tokens = torch.randint(0, vocab_size, (batch_size, seq_length))
    output = processor(tokens)

    assert output.shape == (batch_size, seq_length, embed_dim)

    # Test with attention mask
    mask = torch.zeros(batch_size, seq_length, seq_length)
    mask[:, :, seq_length // 2 :] = float("-inf")  # Mask future positions
    output_masked = processor(tokens, mask)

    assert output_masked.shape == (batch_size, seq_length, embed_dim)
    assert not torch.equal(output, output_masked)

    # Test with shorter sequence
    short_tokens = torch.randint(0, vocab_size, (batch_size, seq_length // 2))
    short_output = processor(short_tokens)

    assert short_output.shape == (batch_size, seq_length // 2, embed_dim)


def test_cross_modal_fusion():
    """Test CrossModalFusion functionality."""
    batch_size = 16
    seq_length = 50

    # Define modality dimensions
    modality_dims = {"vision": 1024, "text": 512, "audio": 256}
    fusion_dim = 768

    fusion = CrossModalFusion(modality_dims=modality_dims, fusion_dim=fusion_dim)

    # Create inputs for each modality
    inputs = {
        "vision": torch.randn(batch_size, seq_length, modality_dims["vision"]),
        "text": torch.randn(batch_size, seq_length, modality_dims["text"]),
        "audio": torch.randn(batch_size, seq_length, modality_dims["audio"]),
    }

    # Test basic fusion
    output = fusion(inputs)
    assert output.shape == (batch_size, seq_length, fusion_dim)

    # Test with masks
    masks = {
        "vision_to_text": torch.zeros(batch_size, seq_length, dtype=torch.bool),
        "vision_to_audio": torch.zeros(batch_size, seq_length, dtype=torch.bool),
        "text_to_vision": torch.zeros(batch_size, seq_length, dtype=torch.bool),
        "text_to_audio": torch.zeros(batch_size, seq_length, dtype=torch.bool),
        "audio_to_vision": torch.zeros(batch_size, seq_length, dtype=torch.bool),
        "audio_to_text": torch.zeros(batch_size, seq_length, dtype=torch.bool),
    }

    # Mask some attention paths
    for mask in masks.values():
        mask[:, seq_length // 2 :] = True

    output_masked = fusion(inputs, masks)
    assert output_masked.shape == (batch_size, seq_length, fusion_dim)
    assert not torch.equal(output, output_masked)

    # Test with subset of modalities
    subset_inputs = {"vision": inputs["vision"], "text": inputs["text"]}
    subset_output = fusion(subset_inputs)
    assert subset_output.shape == (batch_size, seq_length, fusion_dim)


if __name__ == "__main__":
    pytest.main([__file__])
