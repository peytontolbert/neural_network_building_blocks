"""Tests for adaptive and self-improvement components."""

import torch
import torch.nn as nn
import pytest
from nncore.adaptive import (
    AdaptiveComputation,
    MetaLearningModule,
    EvolutionaryLayer,
    PopulationBasedTraining,
)


def test_adaptive_computation():
    """Test AdaptiveComputation functionality."""
    batch_size = 32
    input_dim = 64
    hidden_dim = 128

    layer = AdaptiveComputation(
        input_dim=input_dim, hidden_dim=hidden_dim, max_layers=6, min_layers=2
    )

    # Test forward pass
    x = torch.randn(batch_size, input_dim)
    output, halting_probs, num_steps = layer(x)

    assert output.shape == (batch_size, hidden_dim)
    assert halting_probs.shape[0] == batch_size
    assert halting_probs.shape[2] == 1
    assert num_steps >= layer.min_layers
    assert num_steps <= layer.max_layers

    # Test halting behavior
    layer.eval()
    _, halting_probs2, num_steps2 = layer(x)

    # Should be deterministic in eval mode
    assert torch.allclose(halting_probs, halting_probs2)
    assert num_steps == num_steps2


def test_meta_learning_module():
    """Test MetaLearningModule functionality."""
    batch_size = 16
    input_dim = 32
    hidden_dim = 64
    output_dim = 10

    module = MetaLearningModule(
        input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim
    )

    # Test forward pass
    x = torch.randn(batch_size, input_dim)
    output = module(x)

    assert output.shape == (batch_size, output_dim)

    # Test adaptation
    support_x = torch.randn(batch_size, input_dim)
    support_y = torch.randn(batch_size, output_dim)

    # Store initial outputs
    with torch.no_grad():
        init_output = module(support_x)

    # Store initial parameters of base network
    init_params = []
    for m in module.base_network:
        if isinstance(m, nn.Linear):
            init_params.extend([p.clone() for p in m.parameters()])

    # Perform adaptation
    module.adapt(support_x, support_y, num_adaptation_steps=3)

    # Check base network parameters were updated
    current_params = []
    for m in module.base_network:
        if isinstance(m, nn.Linear):
            current_params.extend([p.clone() for p in m.parameters()])

    params_changed = False
    for p1, p2 in zip(init_params, current_params):
        if not torch.allclose(p1, p2, rtol=1e-3):
            params_changed = True
            break
    assert params_changed, "Base network parameters should change during adaptation"

    # Test adapted forward pass
    with torch.no_grad():
        adapted_output = module(support_x)
    assert not torch.allclose(
        init_output, adapted_output, rtol=1e-3
    ), "Output should change after adaptation"


def test_evolutionary_layer():
    """Test EvolutionaryLayer functionality."""
    batch_size = 24
    input_dim = 48
    hidden_dim = 96

    layer = EvolutionaryLayer(
        input_dim=input_dim, hidden_dim=hidden_dim, num_candidates=5
    )

    # Test forward pass
    x = torch.randn(batch_size, input_dim)
    output = layer(x)

    assert output.shape == (batch_size, hidden_dim)

    # Test evolution
    init_weights = [
        p.clone() for p in layer.population[layer.current_best].parameters()
    ]

    # Perform evolution
    layer.evolve()

    # Check weights were updated
    current_weights = [
        p.clone() for p in layer.population[layer.current_best].parameters()
    ]
    weights_changed = False
    for w1, w2 in zip(init_weights, current_weights):
        if not torch.equal(w1, w2):
            weights_changed = True
            break
    assert weights_changed

    # Test mutation
    candidate = layer.population[0]
    mutated = layer._mutate(candidate)

    assert isinstance(mutated, type(candidate))
    for p1, p2 in zip(candidate.parameters(), mutated.parameters()):
        assert not torch.equal(p1, p2)


def test_population_based_training():
    """Test PopulationBasedTraining functionality."""

    # Define simple model for testing
    class SimpleModel(torch.nn.Module):
        def __init__(self, input_dim=32, hidden_dim=64, output_dim=10):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, output_dim),
            )

        def forward(self, x):
            return self.net(x)

    batch_size = 16
    input_dim = 32
    population_size = 5

    pbt = PopulationBasedTraining(
        model_builder=SimpleModel, population_size=population_size, input_dim=input_dim
    )

    # Test forward pass
    x = torch.randn(batch_size, input_dim)
    outputs = pbt(x)

    assert len(outputs) == population_size
    assert all(out.shape == (batch_size, 10) for out in outputs)

    # Test population update
    init_params = [
        [p.clone() for p in member.parameters()] for member in pbt.population
    ]
    init_hypers = [h.copy() for h in pbt.hyperparameters]

    # Simulate performance metrics
    metrics = [float(i) for i in range(population_size)]
    pbt.update_population(metrics)

    # Check parameters and hyperparameters were updated
    params_changed = False
    hypers_changed = False

    for i in range(population_size):
        current_params = [p.clone() for p in pbt.population[i].parameters()]
        current_hypers = pbt.hyperparameters[i]

        for p1, p2 in zip(init_params[i], current_params):
            if not torch.equal(p1, p2):
                params_changed = True
                break

        if current_hypers != init_hypers[i]:
            hypers_changed = True

        if params_changed and hypers_changed:
            break

    assert params_changed
    assert hypers_changed

    # Test hyperparameter perturbation
    source_hypers = pbt.hyperparameters[0]
    perturbed = pbt._perturb_hyperparameters(source_hypers)

    assert set(perturbed.keys()) == set(source_hypers.keys())
    assert any(perturbed[k] != source_hypers[k] for k in perturbed)

    # Check hyperparameters are within valid ranges
    for name, value in perturbed.items():
        min_val, max_val = pbt.hyper_ranges[name]
        assert min_val <= value <= max_val


if __name__ == "__main__":
    pytest.main([__file__])
