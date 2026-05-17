"""Trainer class for LLM model training. Upon: "Build a Large Language Model (From Scratch)" by Sebastian Raschka, chapter 5."""

import torch
from pyutils.class_util import (get_class, create_instance)
from pyutils.config_util import read_config_arg
from pyutils.kwargs import getKwarg
from llm_utils.dataLoader import createDataLoader
import matplotlib.pyplot as plt

class GPT_Trainer:
    """Trainer class for GPT-style language models.
    
    Handles the complete training pipeline including model optimization, loss computation,
    validation, and checkpointing. Implements training loops with support for distributed
    learning progress tracking and visualization.
    
    Based on: "Build a Large Language Model (From Scratch)" by Sebastian Raschka, chapter 5.
    
    Attributes:
        llm: The language model instance to train.
        cfg: Configuration dictionary containing training hyperparameters and settings.
        device: PyTorch device for computation (CPU or GPU).
        optimizer: Optimizer instance for model weight updates.
    """

    def __init__(self, llm, cfg):
        """Initialize the trainer with a language model and configuration.
        
        Args:
            llm: Language model instance with getModel() and getTokenizer() methods.
            cfg: Configuration dictionary with training parameters.
        """
        self.llm = llm
        self.cfg = cfg
        self.device = torch.device(read_config_arg(self.cfg, "device", "cpu"))
        self.optimizer = self._create_optimizer()

    def _create_optimizer(self):
        """Create the optimizer for model training.
        
        Retrieves optimizer class and hyperparameters from configuration, then
        instantiates it with the model parameters.
        
        Returns:
            torch.optim.Optimizer: Configured optimizer instance.
        """
        model = self.llm.getModel()
        Optimizer = get_class(read_config_arg(self.cfg, "Optimizer", "torch.optim.AdamW"))
        learning_rate = read_config_arg(self.cfg, "learning_rate", 5e-4)
        weight_decay = read_config_arg(self.cfg, "weight_decay", 0.1)
        return Optimizer(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def _calc_loss_batch(self, input_batch, target_batch, model):
        """Calculate cross-entropy loss for a batch.
        
        Moves data to the appropriate device, computes model logits, and calculates
        cross-entropy loss between predictions and targets.
        
        Args:
            input_batch (torch.Tensor): Input token batch.
            target_batch (torch.Tensor): Target token batch.
            model: Language model for generating logits.
            
        Returns:
            torch.Tensor: Scalar loss value.
        """
        input_batch, target_batch = input_batch.to(self.device), target_batch.to(self.device)
        logits = model(input_batch)
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
        return loss

    def _calc_loss_loader(self, data_loader, model, num_batches=None):
        """Calculate average loss over a data loader.
        
        Computes loss across specified number of batches, averaging the results.
        Defaults to all batches if num_batches is not specified.
        
        Args:
            data_loader: PyTorch DataLoader instance.
            model: Language model for inference.
            num_batches (int, optional): Number of batches to evaluate. Defaults to all.
            
        Returns:
            float: Average loss across evaluated batches.
        """
        total_loss = 0.
        if len(data_loader) == 0:
            return float("nan")
        elif num_batches is None:
            num_batches = len(data_loader)
        else:
            num_batches = min(num_batches, len(data_loader))
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i < num_batches:
                loss = self._calc_loss_batch(input_batch, target_batch, model)
                total_loss += loss.item()
            else:
                break
        return total_loss / num_batches

    def _evaluate_model(self, model, train_loader, val_loader, eval_iter):
        """Evaluate model on training and validation datasets.
        
        Sets model to evaluation mode, computes losses without gradient calculation,
        then returns to training mode.
        
        Args:
            model: Language model to evaluate.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            eval_iter (int): Number of batches to evaluate per dataset.
            
        Returns:
            tuple: (train_loss, val_loss) as floats.
        """
        model.eval()
        with torch.no_grad():
            train_loss = self._calc_loss_loader(train_loader, model, num_batches=eval_iter)
            val_loss = self._calc_loss_loader(val_loader, model, num_batches=eval_iter)
        model.train()
        return train_loss, val_loss    

    def _checkpoint(self, model, checkpoint_prompt):
        """Generate and print sample text for model checkpoint evaluation.
        
        Uses the model in evaluation mode to generate text from a prompt, providing
        qualitative feedback during training.
        
        Args:
            model: Language model for text generation.
            checkpoint_prompt (str): Initial prompt text for generation.
        """
        model.eval()
        context_size = model.pos_emb.weight.shape[0]
        encoded, _ = self.llm.text_encode(checkpoint_prompt)
        encoded.to(self.device)
        with torch.no_grad():
            token_ids = self.llm.generate_text_simple(tokens_batch=encoded, max_new_tokens=50, context_size=context_size)
            decoded_text = self.llm.text_decode(token_ids)
            print(decoded_text.replace("\n", " "))  # Compact print format
        model.train()

    def _calcDataSplitIdx(self, data_length):
        """Calculate the split index for train/validation split.
        
        Args:
            data_length (int): Total length of dataset.
            
        Returns:
            int: Index for splitting training and validation data.
        """
        train_ratio = read_config_arg(self.cfg, "train_ratio", 0.9)
        split_idx = int(train_ratio * data_length)
        return split_idx

    def _createTrainLoader(self, text_data):
        """Create training data loader from text data.
        
        Args:
            text_data: Text data to split and load for training.
            
        Returns:
            torch.utils.data.DataLoader: Training data loader.
        """
        split_idx = self._calcDataSplitIdx(len(text_data))
        train_loader = createDataLoader(text_data[:split_idx], self.cfg)
        return train_loader

    def _createValLoader(self, text_data):
        """Create validation data loader from text data.
        
        Args:
            text_data: Text data to split and load for validation.
            
        Returns:
            torch.utils.data.DataLoader: Validation data loader.
        """
        split_idx = self._calcDataSplitIdx(len(text_data))
        val_loader = createDataLoader(text_data[split_idx:], self.cfg)
        return val_loader

    def train_model(self, text_data):
        """Train the language model on provided text data.
        
        Executes the main training loop over multiple epochs, computing losses,
        updating weights, evaluating on validation set, and checkpointing progress.
        
        Args:
            text_data: Text corpus for training.
            
        Returns:
            tuple: (train_losses, val_losses, track_tokens_seen) - lists tracking
                   training progress across evaluation steps.
        """
        num_epochs = read_config_arg(self.cfg, "num_epochs", 10)
        eval_freq = read_config_arg(self.cfg, "eval_freq", 5) 
        eval_iter = read_config_arg(self.cfg, "eval_iter", 1) 
        checkpoint_prompt = read_config_arg(self.cfg, "checkpoint_prompt", "Every effort moves you")                

        model = self.llm.getModel()
        tokenizer = self.llm.getTokenizer()
        optimizer = self.optimizer

        train_loader = self._createTrainLoader(text_data)
        val_loader = self._createValLoader(text_data)
        print(f"Training on {len(train_loader.dataset)} samples, validating on {len(val_loader.dataset)} samples.")

        # Initialize lists to track losses and tokens seen
        train_losses, val_losses, track_tokens_seen = [], [], []
        tokens_seen = 0
        global_step = -1

        # Main training loop
        for epoch in range(num_epochs):
            model.train()  # Set model to training mode

            for input_batch, target_batch in train_loader:
                optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
                loss = self._calc_loss_batch(input_batch, target_batch, model) # 'logits = model(input_batch)' inside
                loss.backward()  # Calculate loss gradients
                optimizer.step()  # Update model weights using loss gradients
                tokens_seen += input_batch.numel()
                global_step += 1

                # Optional evaluation step
                if global_step % eval_freq == 0:
                    train_loss, val_loss = self._evaluate_model(model, train_loader, val_loader, eval_iter)
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens_seen.append(tokens_seen)
                    print(f"Ep {epoch+1} (Step {global_step:06d}): Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

            # Print a sample text after each epoch
            self._checkpoint(model, checkpoint_prompt)

        return train_losses, val_losses, track_tokens_seen        

    def plot_losses(self, tokens_seen, train_losses, val_losses, **kwargs):
        """Plot training and validation losses over epochs and tokens seen.
        
        Creates a dual-axis plot showing loss progression against both training epochs
        and total tokens processed.
        
        Args:
            tokens_seen (list): Tokens processed at each evaluation step.
            train_losses (list): Training loss values at each evaluation step.
            val_losses (list): Validation loss values at each evaluation step.
            **kwargs: Optional arguments:
                show (bool): Whether to display the plot. Defaults to False.
                filename (str): Path to save plot image. Defaults to None (no save).
        """
        show = getKwarg(kwargs, "show", False)
        filename = getKwarg(kwargs, "filename", None)

        num_epochs = read_config_arg(self.cfg, "num_epochs", 10)
        epochs_seen = torch.linspace(0, num_epochs, len(train_losses))

        fig, ax1 = plt.subplots()

        # Plot training and validation loss against epochs
        ax1.plot(epochs_seen, train_losses, label="Training loss")
        ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss")
        ax1.legend(loc="upper right")

        # Create a second x-axis for tokens seen
        ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
        ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
        ax2.set_xlabel("Tokens seen")

        fig.tight_layout()  # Adjust layout to make room

        if show:
            plt.show()
        if filename:
            plt.savefig(filename)
