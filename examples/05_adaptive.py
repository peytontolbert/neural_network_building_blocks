"""Example demonstrating the usage of adaptive components from the nncore package."""

import torch
import torch.nn as nn
from src.nncore.adaptive import (
    AdaptiveComputation,
    MetaLearningModule,
    EvolutionaryLayer,
    PopulationBasedTraining
)
from src.nncore.utils import DeviceManager

# Set random seed for reproducibility
torch.manual_seed(42)

def demonstrate_adaptive_computation():
    """Demonstrate AdaptiveComputation functionality."""
    print("\n=== AdaptiveComputation Demo ===")
    
    # Create an AdaptiveComputation layer
    batch_size = 16
    input_dim = 128
    hidden_dim = 256
    
    layer = AdaptiveComputation(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        max_layers=6,
        min_layers=2,
        halting_threshold=0.9
    )
    
    # Generate sample input
    x = torch.randn(batch_size, input_dim)
    
    # Forward pass
    output, halting_probs, num_steps = layer(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Halting probabilities shape: {halting_probs.shape}")
    print(f"Number of computation steps: {num_steps}")
    
    # Test in eval mode
    print("\nTesting in eval mode:")
    layer.eval()
    output_eval, halting_probs_eval, num_steps_eval = layer(x)
    print(f"Steps in eval mode: {num_steps_eval}")
    print(f"Deterministic behavior: {torch.allclose(output, output_eval)}")

def demonstrate_meta_learning():
    """Demonstrate MetaLearningModule functionality."""
    print("\n=== MetaLearningModule Demo ===")
    
    # Create a MetaLearningModule
    batch_size = 32
    input_dim = 64
    hidden_dim = 128
    output_dim = 10
    
    module = MetaLearningModule(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        meta_hidden_dim=64,
        num_meta_layers=3
    )
    
    # Generate sample data
    x = torch.randn(batch_size, input_dim)
    
    # Initial forward pass
    output = module(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test adaptation
    print("\nTesting adaptation:")
    support_x = torch.randn(batch_size, input_dim)
    support_y = torch.randn(batch_size, output_dim)
    
    # Store initial parameters
    init_params = [p.clone() for p in module.parameters()]
    
    # Perform adaptation
    module.adapt(support_x, support_y, num_adaptation_steps=3)
    
    # Check parameter updates
    params_changed = False
    for p1, p2 in zip(init_params, module.parameters()):
        if not torch.equal(p1, p2):
            params_changed = True
            break
    print(f"Parameters updated: {params_changed}")
    
    # Test adapted forward pass
    output_adapted = module(x)
    print(f"Adaptation changed output: {not torch.equal(output, output_adapted)}")

def demonstrate_evolutionary_layer():
    """Demonstrate EvolutionaryLayer functionality."""
    print("\n=== EvolutionaryLayer Demo ===")
    
    # Create an EvolutionaryLayer
    batch_size = 24
    input_dim = 64
    hidden_dim = 128
    
    layer = EvolutionaryLayer(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_candidates=5,
        tournament_size=3
    )
    
    # Generate sample input
    x = torch.randn(batch_size, input_dim)
    
    # Initial forward pass
    output = layer(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test evolution
    print("\nTesting evolution:")
    init_weights = [p.clone() for p in layer.population[layer.current_best].parameters()]
    
    # Perform evolution
    layer.evolve()
    
    # Check weight updates
    current_weights = [p.clone() for p in layer.population[layer.current_best].parameters()]
    weights_changed = False
    for w1, w2 in zip(init_weights, current_weights):
        if not torch.equal(w1, w2):
            weights_changed = True
            break
    print(f"Weights evolved: {weights_changed}")
    
    # Test mutation
    print("\nTesting mutation:")
    candidate = layer.population[0]
    mutated = layer._mutate(candidate)
    
    mutation_diff = 0
    for p1, p2 in zip(candidate.parameters(), mutated.parameters()):
        mutation_diff += (p1 - p2).abs().mean().item()
    print(f"Average mutation difference: {mutation_diff:.6f}")

def demonstrate_population_based_training():
    """Demonstrate PopulationBasedTraining functionality."""
    print("\n=== PopulationBasedTraining Demo ===")
    
    # Define simple model for testing
    class SimpleModel(nn.Module):
        def __init__(self, input_dim=32, hidden_dim=64, output_dim=10):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )
        
        def forward(self, x):
            return self.net(x)
    
    # Create PopulationBasedTraining
    batch_size = 16
    input_dim = 32
    population_size = 5
    
    pbt = PopulationBasedTraining(
        model_builder=SimpleModel,
        population_size=population_size,
        input_dim=input_dim
    )
    
    # Generate sample input
    x = torch.randn(batch_size, input_dim)
    
    # Forward pass
    outputs = pbt(x)
    print(f"Input shape: {x.shape}")
    print(f"Number of population outputs: {len(outputs)}")
    print(f"Individual output shape: {outputs[0].shape}")
    
    # Test population update
    print("\nTesting population update:")
    init_params = [[p.clone() for p in member.parameters()] for member in pbt.population]
    init_hypers = [h.copy() for h in pbt.hyperparameters]
    
    # Simulate performance metrics
    metrics = [float(i) for i in range(population_size)]
    pbt.update_population(metrics)
    
    # Check updates
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
    
    print(f"Parameters updated: {params_changed}")
    print(f"Hyperparameters updated: {hypers_changed}")

class AdaptiveNetwork(nn.Module):
    """Example network combining different adaptive components."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_evolutionary_layers: int = 3
    ):
        super().__init__()
        
        # Adaptive computation layer
        self.adaptive = AdaptiveComputation(
            input_dim=input_dim,
            hidden_dim=hidden_dim
        )
        
        # Meta-learning module
        self.meta = MetaLearningModule(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim
        )
        
        # Evolutionary layers
        self.evolutionary = nn.ModuleList([
            EvolutionaryLayer(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim
            ) for _ in range(num_evolutionary_layers)
        ])
        
        # Output layer
        self.output = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # Adaptive computation
        x, _, _ = self.adaptive(x)
        
        # Meta-learning adaptation
        x = self.meta(x)
        
        # Evolutionary processing
        for layer in self.evolutionary:
            x = layer(x)
        
        # Output
        return self.output(x)
    
    def adapt(self, support_x, support_y):
        """Adapt the network using support data."""
        # Meta-learning adaptation
        self.meta.adapt(support_x, support_y)
        
        # Evolve layers
        for layer in self.evolutionary:
            layer.evolve()

def demonstrate_adaptive_network():
    """Demonstrate the complete adaptive network."""
    print("\n=== AdaptiveNetwork Demo ===")
    
    # Create network
    batch_size = 16
    input_dim = 64
    hidden_dim = 128
    output_dim = 10
    
    network = AdaptiveNetwork(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim
    )
    
    # Generate sample data
    x = torch.randn(batch_size, input_dim)
    support_x = torch.randn(batch_size, input_dim)
    support_y = torch.randn(batch_size, output_dim)
    
    # Initial forward pass
    output = network(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test adaptation
    print("\nTesting adaptation:")
    network.adapt(support_x, support_y)
    output_adapted = network(x)
    print(f"Adaptation changed output: {not torch.equal(output, output_adapted)}")
    
    # Test with device management
    network = DeviceManager.to_device(network)
    x = DeviceManager.to_device(x)
    
    output = network(x)
    print(f"\nDevice management:")
    print(f"Model device: {next(network.parameters()).device}")
    print(f"Input device: {x.device}")
    print(f"Output device: {output.device}")

def main():
    """Run all demonstrations."""
    demonstrate_adaptive_computation()
    demonstrate_meta_learning()
    demonstrate_evolutionary_layer()
    demonstrate_population_based_training()
    demonstrate_adaptive_network()

if __name__ == "__main__":
    main() 