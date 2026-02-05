import logging
from pathlib import Path
import torch
from typing import Callable, Optional, Tuple
import matplotlib.pyplot as plt

# Set up logging
log = logging.getLogger(__name__)


class SimpleGA:
    def __init__(
        self,
        fitness_fn: Callable[[torch.Tensor], torch.Tensor],
        dim: int,
        bounds: Tuple[float, float] = (0.0, 1.0),
        pop_size: int = 100,
        cx_prob: float = 0.8,
        mut_prob: float = 0.2,
        mut_scale: float = 0.1,
        tournament_size: int = 3,
        elitism: bool = True,
        crossover_type: str = "uniform",  # "uniform", "sbx", "blx", "arithmetic"
        mutation_type: str = "gaussian",  # "gaussian", "polynomial", "adaptive", "cauchy"
        eta_crossover: float = 15.0,  # SBX parameter
        eta_mutation: float = 20.0,  # Polynomial mutation parameter
        alpha_blx: float = 0.5,  # BLX crossover parameter
        device: str = "cpu",
    ):
        """
        Simple genetic algorithm implementation using PyTorch tensors.

        Args:
            fitness_fn: Function that takes (pop_size, dim) tensor and returns (pop_size,) fitness tensor
            dim: Dimensionality of each individual
            bounds: (min, max) bounds for each gene
            pop_size: Population size
            cx_prob: Crossover probability
            mut_prob: Mutation probability per gene
            mut_scale: Standard deviation for Gaussian mutation
            tournament_size: Size of tournament for selection
            elitism: Whether to keep the best individuals from the previous generation
            crossover_type: Type of crossover ("uniform", "sbx", "blx", "arithmetic")
            mutation_type: Type of mutation ("gaussian", "polynomial", "adaptive", "cauchy")
            eta_crossover: Distribution index for SBX crossover
            eta_mutation: Distribution index for polynomial mutation
            alpha_blx: Alpha parameter for BLX crossover
            device: "cpu", "cuda", or "mps" for computation
        """
        self.fitness_fn = fitness_fn
        self.dim = dim
        self.bounds = bounds
        self.pop_size = pop_size
        self.cx_prob = cx_prob
        self.mut_prob = mut_prob
        self.mut_scale = mut_scale
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.crossover_type = crossover_type
        self.mutation_type = mutation_type
        self.eta_crossover = eta_crossover
        self.eta_mutation = eta_mutation
        self.alpha_blx = alpha_blx
        self.device = device

        # Initialize population randomly within bounds
        self.population = self._random_population()
        self.generation = 0
        self.best_fitness_history = []

    def _random_population(self) -> torch.Tensor:
        """Generate random population within bounds."""
        low, high = self.bounds
        return (
            torch.rand(self.pop_size, self.dim, device=self.device) * (high - low) + low
        )

    def _tournament_selection(self, fitness: torch.Tensor) -> torch.Tensor:
        """Vectorized tournament selection."""
        candidates = torch.randint(
            0, self.pop_size, (self.pop_size, self.tournament_size), device=self.device
        )
        tournament_fitness = fitness[candidates]
        winners = torch.argmax(tournament_fitness, dim=1)
        winner_indices = candidates[torch.arange(self.pop_size), winners]
        return self.population[winner_indices]

    # CROSSOVER METHODS
    def _uniform_crossover(self, parents: torch.Tensor) -> torch.Tensor:
        """Apply uniform crossover to create offspring."""
        offspring = parents.clone()

        for i in range(0, self.pop_size - 1, 2):
            if torch.rand(1, device=self.device).item() < self.cx_prob:
                mask = torch.rand(self.dim, device=self.device) < 0.5
                parent_a, parent_b = offspring[i].clone(), offspring[i + 1].clone()
                offspring[i][mask] = parent_b[mask]
                offspring[i + 1][mask] = parent_a[mask]

        return offspring

    def _sbx_crossover(self, parents: torch.Tensor) -> torch.Tensor:
        """Simulated Binary Crossover - best for continuous spaces."""
        offspring = parents.clone()
        low, high = self.bounds

        for i in range(0, self.pop_size - 1, 2):
            if torch.rand(1, device=self.device).item() < self.cx_prob:
                parent1, parent2 = offspring[i], offspring[i + 1]

                # Random values for each dimension
                u = torch.rand(self.dim, device=self.device)

                # Calculate beta values
                beta = torch.where(
                    u <= 0.5,
                    (2 * u) ** (1.0 / (self.eta_crossover + 1)),
                    (1.0 / (2 * (1 - u))) ** (1.0 / (self.eta_crossover + 1)),
                )

                # Create offspring
                child1 = 0.5 * ((1 + beta) * parent1 + (1 - beta) * parent2)
                child2 = 0.5 * ((1 - beta) * parent1 + (1 + beta) * parent2)

                offspring[i] = child1
                offspring[i + 1] = child2

        return torch.clamp(offspring, low, high)

    def _blx_crossover(self, parents: torch.Tensor) -> torch.Tensor:
        """Blend crossover - good for exploration."""
        offspring = parents.clone()
        low, high = self.bounds

        for i in range(0, self.pop_size - 1, 2):
            if torch.rand(1, device=self.device).item() < self.cx_prob:
                p1, p2 = offspring[i], offspring[i + 1]

                # Find min/max for each dimension
                cmin = torch.min(p1, p2)
                cmax = torch.max(p1, p2)

                # Extend range by alpha
                d = cmax - cmin
                range_low = cmin - self.alpha_blx * d
                range_high = cmax + self.alpha_blx * d

                # Generate offspring uniformly in extended range
                offspring[i] = (
                    torch.rand(self.dim, device=self.device) * (range_high - range_low)
                    + range_low
                )
                offspring[i + 1] = (
                    torch.rand(self.dim, device=self.device) * (range_high - range_low)
                    + range_low
                )

        return torch.clamp(offspring, low, high)

    def _arithmetic_crossover(self, parents: torch.Tensor) -> torch.Tensor:
        """Weighted average crossover - conservative but stable."""
        offspring = parents.clone()

        for i in range(0, self.pop_size - 1, 2):
            if torch.rand(1, device=self.device).item() < self.cx_prob:
                # Random weight for each pair
                alpha = torch.rand(1, device=self.device)

                child1 = alpha * offspring[i] + (1 - alpha) * offspring[i + 1]
                child2 = (1 - alpha) * offspring[i] + alpha * offspring[i + 1]

                offspring[i] = child1
                offspring[i + 1] = child2

        return offspring

    # MUTATION METHODS
    def _gaussian_mutation(self, offspring: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian mutation to offspring."""
        mutation_mask = (
            torch.rand(self.pop_size, self.dim, device=self.device) < self.mut_prob
        )

        noise = (
            torch.randn(self.pop_size, self.dim, device=self.device) * self.mut_scale
        )
        offspring += mutation_mask.float() * noise

        low, high = self.bounds
        offspring = torch.clamp(offspring, low, high)
        return offspring

    def _polynomial_mutation(self, offspring: torch.Tensor) -> torch.Tensor:
        """Polynomial mutation - better than Gaussian for bounded continuous spaces."""

        mutation_mask = (
            torch.rand(self.pop_size, self.dim, device=self.device) < self.mut_prob
        )

        for i in range(self.pop_size):
            for j in range(self.dim):
                if mutation_mask[i, j]:
                    y = offspring[i, j]
                    ll, hh = self.bounds
                    low = ll[j]
                    high = hh[j]

                    # Normalize to [0,1]
                    yl = (y - low) / (high - low)
                    yu = (high - y) / (high - low)

                    rand = torch.rand(1, device=self.device).item()

                    if rand <= 0.5:
                        xy = 1.0 - yl
                        val = 2.0 * rand + (1.0 - 2.0 * rand) * (
                            xy ** (self.eta_mutation + 1)
                        )
                        deltaq = (val ** (1.0 / (self.eta_mutation + 1))) - 1.0
                    else:
                        xy = 1.0 - yu
                        val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (
                            xy ** (self.eta_mutation + 1)
                        )
                        deltaq = 1.0 - (val ** (1.0 / (self.eta_mutation + 1)))

                    # Apply mutation
                    offspring[i, j] = y + deltaq * (high - low)

        return torch.clamp(offspring, low, high)

    def _adaptive_gaussian_mutation(self, offspring: torch.Tensor) -> torch.Tensor:
        """Gaussian mutation with adaptive step size."""
        # Calculate population diversity to adapt mutation strength
        pop_std = torch.std(self.population, dim=0)
        mean_std = torch.mean(pop_std)
        adaptive_scale = self.mut_scale * (1.0 + pop_std / (mean_std + 1e-8))

        mutation_mask = (
            torch.rand(self.pop_size, self.dim, device=self.device) < self.mut_prob
        )

        # Different noise scale per dimension
        noise = (
            torch.randn(self.pop_size, self.dim, device=self.device) * adaptive_scale
        )
        offspring += mutation_mask.float() * noise

        low, high = self.bounds
        return torch.clamp(offspring, low, high)

    def _cauchy_mutation(self, offspring: torch.Tensor) -> torch.Tensor:
        """Cauchy mutation - heavier tails than Gaussian, better exploration."""
        mutation_mask = (
            torch.rand(self.pop_size, self.dim, device=self.device) < self.mut_prob
        )

        # Cauchy distribution (heavier tails)
        u = torch.rand(self.pop_size, self.dim, device=self.device)
        cauchy_noise = torch.tan(torch.pi * (u - 0.5)) * self.mut_scale

        offspring += mutation_mask.float() * cauchy_noise
        low, high = self.bounds
        return torch.clamp(offspring, low, high)

    def _crossover(self, parents: torch.Tensor) -> torch.Tensor:
        """Apply the selected crossover method."""
        if self.crossover_type == "sbx":
            return self._sbx_crossover(parents)
        elif self.crossover_type == "blx":
            return self._blx_crossover(parents)
        elif self.crossover_type == "arithmetic":
            return self._arithmetic_crossover(parents)
        else:  # uniform
            return self._uniform_crossover(parents)

    def _mutation(self, offspring: torch.Tensor) -> torch.Tensor:
        """Apply the selected mutation method."""
        if self.mutation_type == "polynomial":
            return self._polynomial_mutation(offspring)
        elif self.mutation_type == "adaptive":
            return self._adaptive_gaussian_mutation(offspring)
        elif self.mutation_type == "cauchy":
            return self._cauchy_mutation(offspring)
        else:  # gaussian
            return self._gaussian_mutation(offspring)

    def evolve_generation(self) -> float:
        """Evolve one generation and return best fitness."""
        # Evaluate fitness
        fitness = self.fitness_fn(self.population)

        # Track best fitness
        best_fitness = torch.max(fitness).item()
        self.best_fitness_history.append(best_fitness)

        # Store elite individuals (top 5% or at least 1)
        n_elite = max(1, self.pop_size // 20)
        elite_indices = torch.topk(fitness, n_elite).indices
        self.elite_individuals = self.population[elite_indices].clone()
        self.elite_fitness = fitness[elite_indices].clone()

        # Selection
        parents = self._tournament_selection(fitness)

        # Crossover
        offspring = self._crossover(parents)

        # Mutation
        offspring = self._mutation(offspring)

        # Randomly mix in the most elite individuals from the previous generation
        if self.elitism:
            rand_indices = torch.randperm(self.pop_size, device=self.device)[:n_elite]
            offspring[rand_indices] = self.elite_individuals

        # Replace population
        self.population = offspring
        self.generation += 1

        return best_fitness

    def plot_outcome(self, figure_dir: Optional[Path] = None):
        if figure_dir is not None:
            figure_dir.mkdir(parents=True, exist_ok=True)
            # Plot best fitness history
            plt.figure(figsize=(10, 5))
            plt.plot(self.best_fitness_history, label="Best Fitness")
            plt.xlabel("Generation")
            plt.ylabel("Fitness")
            plt.title("Best Fitness Over Generations")
            plt.legend()
            plt.savefig(
                figure_dir / "best_fitness_history.png", dpi=150, bbox_inches="tight"
            )
            plt.close()

            # Plot final population pairplots of each feature
            # Convert population to numpy for plotting
            pop_array = self.population.cpu().numpy()

            # Create pairplot using matplotlib
            n_dims = self.dim
            fig, axes = plt.subplots(n_dims, n_dims, figsize=(2 * n_dims, 2 * n_dims))

            for i in range(n_dims):
                for j in range(n_dims):
                    ax = axes[i, j] if n_dims > 1 else axes

                    if i == j:
                        # Diagonal: histogram
                        ax.hist(pop_array[:, i], bins=20, alpha=0.7, density=True)
                        ax.set_ylabel("Density")
                    else:
                        # Off-diagonal: scatter plot
                        ax.scatter(pop_array[:, j], pop_array[:, i], alpha=0.6, s=10)
                        ax.set_ylabel(f"Dimension {i}")

                    if i == n_dims - 1:
                        ax.set_xlabel(f"Dimension {j}")
                    else:
                        ax.set_xticklabels([])

            plt.tight_layout()
            plt.savefig(
                figure_dir / "final_population_pairplots.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()

    def run(
        self,
        n_generations: int,
        verbose: bool = True,
        figure_dir: Optional[Path] = None,
    ) -> Tuple[torch.Tensor, float]:
        """
        Run the genetic algorithm for n_generations.

        Args:
            n_generations: Number of generations to evolve
            verbose: Whether to print progress
            figure_dir: Directory to save figures (if any)

        Returns:
            best_individual: Best individual found (tensor)
            best_fitness: Fitness of best individual (float)
        """
        log.info(
            f"Starting GA with {self.pop_size} individuals for {n_generations} generations."
        )
        for gen in range(n_generations):
            best_fitness = self.evolve_generation()

            if verbose and gen % 10 == 0:
                log.info(f"Generation {gen}: Best fitness = {best_fitness:.6f}")

        # Return best individual from final population
        final_fitness = self.fitness_fn(self.population)
        best_idx = torch.argmax(final_fitness)

        log.info(f"GA completed. Best fitness: {final_fitness[best_idx].item():.6f}")
        self.plot_outcome(figure_dir)

        return self.population[best_idx], final_fitness[best_idx].item()
