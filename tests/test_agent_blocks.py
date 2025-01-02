"""Tests for agent-oriented neural network components."""

import torch
import pytest
from nncore.agent_blocks import (
    MemoryModule,
    DecisionLayer,
    StateEncoder,
    PolicyNetwork,
    ValueEstimator
)

def test_memory_module():
    """Test MemoryModule functionality."""
    batch_size = 16
    memory_size = 32
    memory_dim = 64
    query_dim = 128
    
    memory = MemoryModule(
        memory_size=memory_size,
        memory_dim=memory_dim,
        query_dim=query_dim
    )
    
    # Test basic retrieval
    query = torch.randn(batch_size, query_dim)
    retrieved, attention = memory(query)
    
    assert retrieved.shape == (batch_size, memory_dim)
    assert attention.shape[1] == memory.attention.num_heads
    
    # Test with mask
    mask = torch.zeros(batch_size, memory_size, dtype=torch.bool)
    mask[:, memory_size//2:] = True  # Mask second half
    retrieved_masked, attention_masked = memory(query, mask)
    
    assert retrieved_masked.shape == (batch_size, memory_dim)
    assert (attention_masked[:, :, :, memory_size//2:] == 0).all()
    
    # Test memory update
    memory.train()
    _ = memory(query)  # Should update memory
    assert not torch.equal(memory.memory.data, torch.zeros_like(memory.memory))

def test_decision_layer():
    """Test DecisionLayer functionality."""
    batch_size = 32
    input_dim = 64
    output_dim = 10
    
    layer = DecisionLayer(
        input_dim=input_dim,
        output_dim=output_dim
    )
    
    x = torch.randn(batch_size, input_dim)
    
    # Test deterministic output
    layer.eval()
    output = layer(x)
    assert output.shape == (batch_size, output_dim)
    
    # Test with distribution
    layer.train()
    decision, mean, log_var = layer(x, return_distribution=True)
    assert decision.shape == (batch_size, output_dim)
    assert mean.shape == (batch_size, output_dim)
    assert log_var.shape == (batch_size, output_dim)
    
    # Test sampling consistency
    std = torch.exp(0.5 * log_var)
    assert torch.all((decision - mean).abs() <= 3 * std)  # 3-sigma rule

def test_state_encoder():
    """Test StateEncoder functionality."""
    batch_size = 16
    seq_length = 20
    input_dim = 32
    hidden_dim = 64
    
    encoder = StateEncoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim
    )
    
    x = torch.randn(batch_size, seq_length, input_dim)
    output = encoder(x)
    
    assert output.shape == (batch_size, seq_length, hidden_dim)
    
    # Test with attention mask
    mask = torch.zeros(batch_size, seq_length, seq_length)
    mask[:, :, seq_length//2:] = float('-inf')  # Mask future positions
    output_masked = encoder(x, mask)
    
    assert output_masked.shape == (batch_size, seq_length, hidden_dim)
    assert not torch.equal(output, output_masked)

def test_policy_network():
    """Test PolicyNetwork functionality."""
    batch_size = 32
    state_dim = 64
    action_dim = 10
    
    # Test discrete policy
    discrete_policy = PolicyNetwork(
        state_dim=state_dim,
        action_dim=action_dim,
        discrete=True
    )
    
    state = torch.randn(batch_size, state_dim)
    
    # Test deterministic mode
    action_probs = discrete_policy(state, deterministic=True)
    assert action_probs.shape == (batch_size, action_dim)
    assert torch.allclose(action_probs.sum(dim=-1), torch.ones(batch_size))
    
    # Test sampling mode
    dist = discrete_policy(state, deterministic=False)
    assert isinstance(dist, torch.distributions.Categorical)
    
    # Test continuous policy
    continuous_policy = PolicyNetwork(
        state_dim=state_dim,
        action_dim=action_dim,
        discrete=False
    )
    
    # Test deterministic mode
    mean_action = continuous_policy(state, deterministic=True)
    assert mean_action.shape == (batch_size, action_dim)
    
    # Test sampling mode
    action, log_prob = continuous_policy(state, deterministic=False)
    assert action.shape == (batch_size, action_dim)
    assert log_prob.shape == (batch_size,)

def test_value_estimator():
    """Test ValueEstimator functionality."""
    batch_size = 32
    state_dim = 64
    ensemble_size = 5
    
    estimator = ValueEstimator(
        state_dim=state_dim,
        ensemble_size=ensemble_size
    )
    
    state = torch.randn(batch_size, state_dim)
    mean_value, std_value = estimator(state)
    
    assert mean_value.shape == (batch_size, 1)
    assert std_value.shape == (batch_size, 1)
    
    # Test ensemble diversity
    values = torch.stack([
        estimator.value_heads[i](estimator.feature_extractor(state))
        for i in range(ensemble_size)
    ])
    assert not torch.allclose(values[0], values[1])  # Different ensemble members should give different predictions

if __name__ == "__main__":
    pytest.main([__file__]) 