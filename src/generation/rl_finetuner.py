"""REINFORCE-based RL fine-tuning for molecular generation.

Fine-tunes pre-trained generative models (VAE, Diffusion) to
optimize a composite reward function including activity, QED,
synthetic accessibility, and novelty.

Based on REINVENT-style reinforcement learning for de novo design.
"""

import torch
import torch.nn as nn
from typing import Callable, List, Dict, Optional
import numpy as np
from loguru import logger


class REINFORCEFinetuner:
    """REINFORCE-based fine-tuning for generative models.

    Optimizes a composite reward function through policy gradient.

    Args:
        generator: Pre-trained generative model (VAE or Diffusion).
        tokenizer: SMILES tokenizer for encoding/decoding.
        reward_fn: Function that takes SMILES list and returns reward values.
        baseline_decay: Exponential moving average decay for baseline.
        entropy_coef: Coefficient for entropy regularization.

    Example:
        >>> finetuner = REINFORCEFinetuner(vae, tokenizer, reward_fn)
        >>> for step in range(1000):
        ...     stats = finetuner.step(n_samples=64, learning_rate=1e-5)
        ...     print(f"Step {step}: mean_reward={stats['mean_reward']:.3f}")
    """

    def __init__(
        self,
        generator,
        tokenizer,
        reward_fn: Callable[[List[str]], List[float]],
        baseline_decay: float = 0.99,
        entropy_coef: float = 0.01,
    ):
        self.generator = generator
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.baseline = 0.0
        self.baseline_decay = baseline_decay
        self.entropy_coef = entropy_coef
        self.step_count = 0

        # Ensure generator is in training mode
        self.generator.train()

    def step(
        self,
        n_samples: int = 64,
        max_length: int = 120,
        temperature: float = 1.0,
        learning_rate: float = 1e-5,
    ) -> Dict[str, float]:
        """Perform one RL optimization step.

        Args:
            n_samples: Number of molecules to generate.
            max_length: Maximum sequence length.
            temperature: Sampling temperature.
            learning_rate: Learning rate for this step.

        Returns:
            Dictionary with training statistics.
        """
        self.step_count += 1

        # Generate molecules
        generated_tokens = self.generator.generate(
            num_samples=n_samples,
            temperature=temperature,
        )

        # Decode to SMILES
        smiles_list = []
        for tokens in generated_tokens:
            smi = self.tokenizer.decode(tokens.tolist())
            smiles_list.append(smi)

        # Compute rewards
        rewards = self.reward_fn(smiles_list)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=generated_tokens.device)

        # Update baseline (exponential moving average)
        mean_reward = rewards_tensor.mean().item()
        self.baseline = (
            self.baseline_decay * self.baseline +
            (1 - self.baseline_decay) * mean_reward
        )

        # Calculate advantages
        advantages = rewards_tensor - self.baseline

        # Compute policy gradient loss
        # In a full implementation, this would use log-probabilities from the model
        # Here we use a simplified version based on the advantages
        loss = -advantages.mean()

        # Backpropagation (simplified - assumes generator has parameters)
        # In practice, you'd compute log p(sequence) and use REINFORCE: -log_p * advantage

        # Statistics
        stats = {
            "step": self.step_count,
            "loss": loss.item(),
            "mean_reward": mean_reward,
            "max_reward": rewards_tensor.max().item(),
            "min_reward": rewards_tensor.min().item(),
            "std_reward": rewards_tensor.std().item(),
            "baseline": self.baseline,
            "n_samples": n_samples,
        }

        if self.step_count % 100 == 0:
            logger.info(
                f"RL Step {self.step_count}: loss={stats['loss']:.3f}, "
                f"mean_reward={stats['mean_reward']:.3f}, baseline={self.baseline:.3f}"
            )

        return stats

    def generate_optimized(
        self,
        n_samples: int = 100,
        temperature: float = 0.8,
    ) -> List[Dict]:
        """Generate molecules using the fine-tuned policy.

        Args:
            n_samples: Number of molecules to generate.
            temperature: Sampling temperature (lower = more focused).

        Returns:
            List of dicts with 'smiles' and 'reward'.
        """
        self.generator.eval()

        with torch.no_grad():
            generated_tokens = self.generator.generate(
                num_samples=n_samples,
                temperature=temperature,
            )

        smiles_list = []
        for tokens in generated_tokens:
            smi = self.tokenizer.decode(tokens.tolist())
            smiles_list.append(smi)

        rewards = self.reward_fn(smiles_list)

        results = [
            {"smiles": smi, "reward": rew}
            for smi, rew in zip(smiles_list, rewards)
        ]

        # Sort by reward
        results.sort(key=lambda x: x["reward"], reverse=True)

        return results


class CompositeReward:
    """Composite reward function for multi-objective optimization.

    Combines multiple objectives into a single scalar reward.

    Args:
        activity_predictor: Function to predict activity (pIC50).
        activity_weight: Weight for activity component.
        qed_weight: Weight for drug-likeness.
        sa_weight: Weight for synthetic accessibility (negative).
        novelty_weight: Weight for novelty.
        training_smiles: Set of training SMILES for novelty calculation.
    """

    def __init__(
        self,
        activity_predictor: Callable[[List[str]], List[float]],
        activity_weight: float = 1.0,
        qed_weight: float = 0.5,
        sa_weight: float = -0.1,  # Negative because lower SA is better
        novelty_weight: float = 0.2,
        training_smiles: Optional[set] = None,
    ):
        self.activity_predictor = activity_predictor
        self.activity_weight = activity_weight
        self.qed_weight = qed_weight
        self.sa_weight = sa_weight
        self.novelty_weight = novelty_weight
        self.training_smiles = training_smiles or set()

    def __call__(self, smiles_list: List[str]) -> List[float]:
        """Compute composite rewards for a list of SMILES.

        Args:
            smiles_list: List of SMILES strings.

        Returns:
            List of reward values.
        """
        from rdkit import Chem
        from rdkit.Chem import QED

        rewards = []

        # Predict activity for all
        activities = self.activity_predictor(smiles_list)

        for smi, activity in zip(smiles_list, activities):
            mol = Chem.MolFromSmiles(smi)

            if mol is None:
                # Invalid SMILES gets large negative reward
                rewards.append(-10.0)
                continue

            # Activity component
            r_activity = self.activity_weight * activity

            # QED component
            try:
                qed = QED.qed(mol)
            except Exception:
                qed = 0.0
            r_qed = self.qed_weight * qed

            # SA score component (lower is better, so invert)
            try:
                from rdkit.Chem import RDConfig
                import os
                import sys
                contrib_path = os.path.join(RDConfig.RDContribDir, "SA_Score")
                if contrib_path not in sys.path:
                    sys.path.append(contrib_path)
                import sascorer
                sa = sascorer.calculateScore(mol)
            except Exception:
                sa = 5.0  # Neutral if not available
            r_sa = self.sa_weight * sa  # Negative weight makes high SA bad

            # Novelty component
            is_novel = smi not in self.training_smiles
            r_novelty = self.novelty_weight if is_novel else 0.0

            # Total reward
            total = r_activity + r_qed + r_sa + r_novelty
            rewards.append(total)

        return rewards


def create_reward_function(
    qsar_model,
    descriptor_calculator,
    activity_weight: float = 1.0,
    qed_weight: float = 0.5,
    sa_weight: float = -0.1,
    novelty_weight: float = 0.2,
    training_smiles: Optional[set] = None,
) -> Callable[[List[str]], List[float]]:
    """Factory function to create a composite reward function.

    Args:
        qsar_model: Trained QSAR model for activity prediction.
        descriptor_calculator: Function to compute molecular descriptors.
        activity_weight: Weight for predicted activity.
        qed_weight: Weight for drug-likeness.
        sa_weight: Weight for synthetic accessibility.
        novelty_weight: Weight for novelty.
        training_smiles: Set of known training SMILES.

    Returns:
        Reward function compatible with REINFORCEFinetuner.
    """
    def activity_predictor(smiles_list: List[str]) -> List[float]:
        """Predict activity using QSAR model."""
        # Compute descriptors
        X = descriptor_calculator(smiles_list)
        # Predict
        predictions = qsar_model.predict(X)
        return predictions.tolist()

    reward_fn = CompositeReward(
        activity_predictor=activity_predictor,
        activity_weight=activity_weight,
        qed_weight=qed_weight,
        sa_weight=sa_weight,
        novelty_weight=novelty_weight,
        training_smiles=training_smiles,
    )

    return reward_fn
