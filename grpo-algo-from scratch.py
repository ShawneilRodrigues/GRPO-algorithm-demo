"""
Key Components:
1. Response Generation: Generate K responses for each prompt
2. Reward Calculation: Calculate rewards for each response
3. Weight Calculation: Use softmax to convert rewards to weights
4. Loss Calculation: Weighted negative log-likelihood loss
5. Model Update: Backpropagation and parameter updates
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
import math
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GRPOTrainer:
    """
    GRPO (Generalized Reward Preference Optimization) Trainer
    
    This class implements the GRPO algorithm for training language models using
    reward-based preference optimization.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        reward_function: Callable,
        optimizer: torch.optim.Optimizer,
        device: str = "cuda",
        group_size: int = 4,
        temperature: float = 1.0,
        max_new_tokens: int = 256,
        beta: float = 0.1,  # KL penalty coefficient
        clip_reward: bool = True,
        reward_clip_value: float = 5.0,
    ):
        """
        Initialize the GRPO trainer.
        
        Args:
            model: The language model to train
            tokenizer: Tokenizer for the model
            reward_function: Function that calculates rewards for responses
            optimizer: Optimizer for model parameters
            device: Device to run training on
            group_size: Number of responses to generate per prompt (K)
            temperature: Sampling temperature for response generation
            max_new_tokens: Maximum tokens to generate per response
            beta: KL penalty coefficient for regularization
            clip_reward: Whether to clip reward values
            reward_clip_value: Maximum absolute reward value if clipping
        """
        self.model = model
        self.tokenizer = tokenizer
        self.reward_function = reward_function
        self.optimizer = optimizer
        self.device = device
        self.group_size = group_size
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.beta = beta
        self.clip_reward = clip_reward
        self.reward_clip_value = reward_clip_value
        
        # Training statistics
        self.training_stats = {
            'total_steps': 0,
            'average_rewards': [],
            'losses': [],
            'kl_divergences': []
        }
        
    def generate_responses(
        self, 
        prompt_tokens: torch.Tensor, 
        num_responses: Optional[int] = None
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        Generate multiple responses for a given prompt.
        
        Args:
            prompt_tokens: Tokenized prompt
            num_responses: Number of responses to generate (defaults to group_size)
            
        Returns:
            Tuple of (generated_tokens, response_texts)
        """
        if num_responses is None:
            num_responses = self.group_size
            
        self.model.eval()
        with torch.no_grad():
            # Generate responses using sampling
            generated_outputs = self.model.generate(
                prompt_tokens,
                max_new_tokens=self.max_new_tokens,
                num_return_sequences=num_responses,
                do_sample=True,
                temperature=self.temperature,
                top_k=50,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        # Extract only the newly generated tokens (remove prompt)
        generated_tokens = generated_outputs.sequences[:, prompt_tokens.shape[1]:]
        
        # Decode responses
        response_texts = self.tokenizer.batch_decode(
            generated_tokens, 
            skip_special_tokens=True
        )
        
        return generated_tokens, response_texts
    
    def calculate_rewards(
        self, 
        prompts: List[Dict], 
        responses: List[str], 
        ground_truth: Optional[List[str]] = None
    ) -> torch.Tensor:
        """
        Calculate rewards for generated responses.
        
        Args:
            prompts: List of prompt dictionaries
            responses: List of response strings
            ground_truth: Optional ground truth answers
            
        Returns:
            Tensor of rewards for each response
        """
        # Convert responses to the format expected by reward function
        completions = [{"content": response} for response in responses]
        
        # Calculate rewards using the provided reward function
        if ground_truth is not None:
            rewards = self.reward_function(
                prompts=[prompts] * len(responses), 
                completions=completions, 
                answers=ground_truth * len(responses)
            )
        else:
            rewards = self.reward_function(
                prompts=[prompts] * len(responses), 
                completions=completions
            )
        
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        
        # Clip rewards if specified
        if self.clip_reward:
            rewards = torch.clamp(rewards, -self.reward_clip_value, self.reward_clip_value)
        
        return rewards
    
    def calculate_grpo_weights(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Calculate GRPO weights using softmax of rewards.
        
        Args:
            rewards: Tensor of rewards for each response
            
        Returns:
            Softmax weights for each response
        """
        # Apply softmax to convert rewards to weights
        # Higher rewards get higher weights
        weights = F.softmax(rewards, dim=0)
        return weights
    
    def calculate_log_probabilities(
        self, 
        prompt_tokens: torch.Tensor, 
        response_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate log probabilities for generated responses.
        
        Args:
            prompt_tokens: Tokenized prompt
            response_tokens: Tokenized responses
            
        Returns:
            Log probabilities for each response
        """
        self.model.train()
        
        # Concatenate prompt and response
        full_tokens = torch.cat([prompt_tokens, response_tokens], dim=1)
        
        # Forward pass through model
        outputs = self.model(full_tokens)
        logits = outputs.logits
        
        # Extract logits for response tokens only
        response_logits = logits[:, -response_tokens.shape[1]:, :].contiguous()
        
        # Calculate log probabilities
        log_probs = F.log_softmax(response_logits, dim=-1)
        
        # Gather log probabilities for actual tokens
        response_tokens_expanded = response_tokens.unsqueeze(-1)
        token_log_probs = torch.gather(log_probs, -1, response_tokens_expanded).squeeze(-1)
        
        # Sum log probabilities for each sequence (excluding padding)
        mask = (response_tokens != self.tokenizer.pad_token_id).float()
        sequence_log_probs = (token_log_probs * mask).sum(dim=1)
        
        return sequence_log_probs
    
    def calculate_kl_penalty(
        self, 
        current_log_probs: torch.Tensor, 
        reference_log_probs: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate KL divergence penalty between current and reference model.
        
        Args:
            current_log_probs: Log probabilities from current model
            reference_log_probs: Log probabilities from reference model
            
        Returns:
            KL divergence values
        """
        # KL(current || reference) = sum(current * (log(current) - log(reference)))
        kl_div = current_log_probs - reference_log_probs
        return kl_div
    
    def grpo_loss(
        self,
        prompt_tokens: torch.Tensor,
        response_tokens: torch.Tensor,
        weights: torch.Tensor,
        reference_log_probs: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Calculate the GRPO loss function.
        
        Args:
            prompt_tokens: Tokenized prompt
            response_tokens: Tokenized responses  
            weights: GRPO weights from softmax of rewards
            reference_log_probs: Optional reference model log probabilities
            
        Returns:
            Tuple of (loss, metrics_dict)
        """
        # Calculate log probabilities for each response
        log_probs = []
        for i in range(response_tokens.shape[0]):
            response_i = response_tokens[i:i+1]
            log_prob_i = self.calculate_log_probabilities(prompt_tokens, response_i)
            log_probs.append(log_prob_i)
        
        log_probs = torch.cat(log_probs, dim=0)
        
        # GRPO objective: maximize weighted log likelihood
        weighted_log_likelihood = (weights * log_probs).sum()
        
        # Primary loss: negative weighted log likelihood
        loss = -weighted_log_likelihood
        
        # Add KL penalty if reference probabilities provided
        kl_penalty = 0.0
        if reference_log_probs is not None and self.beta > 0:
            kl_div = self.calculate_kl_penalty(log_probs, reference_log_probs)
            kl_penalty = self.beta * kl_div.mean()
            loss += kl_penalty
        
        # Metrics for logging
        metrics = {
            'weighted_log_likelihood': weighted_log_likelihood.item(),
            'kl_penalty': kl_penalty.item() if isinstance(kl_penalty, torch.Tensor) else kl_penalty,
            'average_log_prob': log_probs.mean().item(),
            'weight_entropy': -(weights * torch.log(weights + 1e-8)).sum().item()
        }
        
        return loss, metrics
    
    def train_step(
        self,
        prompt: Dict,
        ground_truth: Optional[str] = None,
        reference_model: Optional[torch.nn.Module] = None
    ) -> Dict:
        """
        Perform a single GRPO training step.
        
        Args:
            prompt: Prompt dictionary with messages
            ground_truth: Optional ground truth answer
            reference_model: Optional reference model for KL penalty
            
        Returns:
            Dictionary with training metrics
        """
        # Tokenize prompt
        prompt_tokens = self.tokenizer.apply_chat_template(
            prompt, 
            add_generation_prompt=True, 
            return_tensors="pt"
        ).to(self.device)
        
        # Step 1: Generate K responses
        response_tokens, response_texts = self.generate_responses(prompt_tokens)
        
        # Step 2: Calculate rewards
        ground_truth_list = [ground_truth] if ground_truth else None
        rewards = self.calculate_rewards(prompt, response_texts, ground_truth_list)
        
        # Step 3: Calculate GRPO weights
        weights = self.calculate_grpo_weights(rewards)
        
        # Step 4: Get reference log probabilities if reference model provided
        reference_log_probs = None
        if reference_model is not None:
            reference_model.eval()
            with torch.no_grad():
                ref_log_probs = []
                for i in range(response_tokens.shape[0]):
                    response_i = response_tokens[i:i+1]
                    full_tokens = torch.cat([prompt_tokens, response_i], dim=1)
                    ref_outputs = reference_model(full_tokens)
                    ref_logits = ref_outputs.logits[:, -response_i.shape[1]:, :]
                    ref_log_probs_i = F.log_softmax(ref_logits, dim=-1)
                    response_i_expanded = response_i.unsqueeze(-1)
                    token_ref_log_probs = torch.gather(ref_log_probs_i, -1, response_i_expanded).squeeze(-1)
                    mask = (response_i != self.tokenizer.pad_token_id).float()
                    sequence_ref_log_probs = (token_ref_log_probs * mask).sum(dim=1)
                    ref_log_probs.append(sequence_ref_log_probs)
                reference_log_probs = torch.cat(ref_log_probs, dim=0)
        
        # Step 5: Calculate loss
        loss, metrics = self.grpo_loss(
            prompt_tokens, 
            response_tokens, 
            weights, 
            reference_log_probs
        )
        
        # Step 6: Backpropagation and optimization
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Update statistics
        self.training_stats['total_steps'] += 1
        self.training_stats['average_rewards'].append(rewards.mean().item())
        self.training_stats['losses'].append(loss.item())
        if 'kl_penalty' in metrics:
            self.training_stats['kl_divergences'].append(metrics['kl_penalty'])
        
        # Add training-specific metrics
        metrics.update({
            'loss': loss.item(),
            'average_reward': rewards.mean().item(),
            'max_reward': rewards.max().item(),
            'min_reward': rewards.min().item(),
            'reward_std': rewards.std().item(),
            'max_weight': weights.max().item(),
            'min_weight': weights.min().item(),
        })
        
        return metrics
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        reference_model: Optional[torch.nn.Module] = None,
        log_interval: int = 10
    ) -> Dict:
        """
        Train for one epoch.
        
        Args:
            dataloader: DataLoader with training data
            reference_model: Optional reference model for KL penalty
            log_interval: How often to log training progress
            
        Returns:
            Dictionary with epoch metrics
        """
        self.model.train()
        epoch_metrics = []
        
        progress_bar = tqdm(dataloader, desc="Training")
        
        for step, batch in enumerate(progress_bar):
            # Extract prompt and ground truth from batch
            prompt = batch['prompt'][0]  # Assuming batch size 1
            ground_truth = batch.get('answer', [None])[0]
            
            # Perform training step
            step_metrics = self.train_step(prompt, ground_truth, reference_model)
            epoch_metrics.append(step_metrics)
            
            # Log progress
            if step % log_interval == 0:
                avg_loss = np.mean([m['loss'] for m in epoch_metrics[-log_interval:]])
                avg_reward = np.mean([m['average_reward'] for m in epoch_metrics[-log_interval:]])
                progress_bar.set_postfix({
                    'loss': f"{avg_loss:.4f}",
                    'reward': f"{avg_reward:.4f}"
                })
        
        # Calculate epoch averages
        epoch_summary = {}
        for key in epoch_metrics[0].keys():
            epoch_summary[f'avg_{key}'] = np.mean([m[key] for m in epoch_metrics])
            
        return epoch_summary
    
    def save_model(self, save_path: str):
        """Save the trained model."""
        if hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(save_path)
        else:
            torch.save(self.model.state_dict(), f"{save_path}/model.pt")
        
        if hasattr(self.tokenizer, 'save_pretrained'):
            self.tokenizer.save_pretrained(save_path)
    
    def get_training_stats(self) -> Dict:
        """Get training statistics."""
        return self.training_stats


def create_grpo_trainer(
    model,
    tokenizer,
    reward_function,
    learning_rate: float = 5e-5,
    device: str = "cuda",
    **kwargs
) -> GRPOTrainer:
    """
    Factory function to create a GRPO trainer with default settings.
    
    Args:
        model: Language model to train
        tokenizer: Tokenizer for the model
        reward_function: Reward function for calculating response quality
        learning_rate: Learning rate for optimization
        device: Device for training
        **kwargs: Additional arguments for GRPOTrainer
        
    Returns:
        Configured GRPOTrainer instance
    """
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_function=reward_function,
        optimizer=optimizer,
        device=device,
        **kwargs
    )
    
    return trainer


# Example usage and testing functions
def demo_grpo_training():
    """
    Demonstration of how to use the GRPO trainer.
    This is a simplified example showing the key components.
    """
    print("GRPO Algorithm Demo")
    print("=" * 50)
    
    # This would normally be your actual model and tokenizer
    print("In a real implementation, you would:")
    print("1. Load your language model and tokenizer")
    print("2. Define your reward function")
    print("3. Prepare your dataset")
    print("4. Create the GRPO trainer")
    print("5. Run training epochs")
    
    print("\nExample training loop structure:")
    print("""
    # Create trainer
    trainer = create_grpo_trainer(
        model=model,
        tokenizer=tokenizer, 
        reward_function=your_reward_function,
        learning_rate=5e-5,
        group_size=4
    )
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_metrics = trainer.train_epoch(dataloader)
        print(f"Epoch {epoch}: {epoch_metrics}")
    
    # Save trained model
    trainer.save_model("./grpo_model")
    """)


if __name__ == "__main__":
    demo_grpo_training()