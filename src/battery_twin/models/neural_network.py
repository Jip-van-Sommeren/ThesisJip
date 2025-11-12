"""
Neural Network Module - PyTorch Implementation

Implements a feedforward neural network for battery capacity prediction
and residual learning in the hybrid digital twin architecture.

Architecture:
- Input layer: Variable dimension based on features
- Hidden layers: [64, 32, 16] with ReLU activation
- Output layer: 1 neuron (regression)
- Dropout: Configurable (default 0.1) for regularization
- Batch normalization: Optional

Features:
- Training with validation split
- Early stopping and learning rate scheduling
- Model persistence (save/load)
- MC Dropout for uncertainty estimation
- L2 regularization (weight decay)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging
import json

logger = logging.getLogger(__name__)


@dataclass
class NeuralNetConfig:
    """
    Configuration for neural network.

    Attributes:
        hidden_layers: List of hidden layer sizes (default: [64, 32, 16])
        dropout_rate: Dropout probability (default: 0.1)
        learning_rate: Initial learning rate (default: 0.001)
        batch_size: Batch size for training (default: 32)
        epochs: Maximum number of training epochs (default: 100)
        validation_split: Fraction of data for validation (default: 0.2)
        early_stopping_patience: Epochs to wait before stopping (default: 10)
        weight_decay: L2 regularization coefficient (default: 1e-4)
        lr_scheduler_patience: Patience for LR scheduler (default: 5)
        lr_scheduler_factor: Factor to reduce LR (default: 0.5)
        device: Device to use ('cuda' or 'cpu', auto-detect if None)
    """
    hidden_layers: List[int] = None
    dropout_rate: float = 0.1
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    weight_decay: float = 1e-4
    lr_scheduler_patience: int = 5
    lr_scheduler_factor: float = 0.5
    device: Optional[str] = None

    def __post_init__(self):
        """Validate and set defaults."""
        if self.hidden_layers is None:
            self.hidden_layers = [64, 32, 16]

        if not (0 <= self.dropout_rate < 1):
            raise ValueError(f"Dropout rate must be in [0, 1), got {self.dropout_rate}")

        if self.learning_rate <= 0:
            raise ValueError(f"Learning rate must be positive, got {self.learning_rate}")

        if self.batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got {self.batch_size}")

        # Auto-detect device if not specified
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        logger.info(f"Using device: {self.device}")


class FeedforwardNN(nn.Module):
    """
    Feedforward neural network for regression.

    Architecture:
        Input -> Hidden[0] -> Dropout -> Hidden[1] -> Dropout -> ... -> Output

    Each hidden layer uses:
        - Linear transformation
        - ReLU activation
        - Dropout (if dropout_rate > 0)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers: List[int],
        dropout_rate: float = 0.1
    ):
        """
        Initialize network.

        Args:
            input_dim: Number of input features
            hidden_layers: List of hidden layer sizes
            dropout_rate: Dropout probability
        """
        super(FeedforwardNN, self).__init__()

        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate

        # Build layers
        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_layers):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            # ReLU activation
            layers.append(nn.ReLU())
            # Dropout
            if dropout_rate > 0:
                layers.append(nn.Dropout(p=dropout_rate))

            prev_dim = hidden_dim

        # Output layer (no activation for regression)
        layers.append(nn.Linear(prev_dim, 1))

        # Create sequential model
        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

        logger.info(
            f"Created FeedforwardNN: input_dim={input_dim}, "
            f"hidden={hidden_layers}, dropout={dropout_rate}"
        )

    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, 1)
        """
        return self.network(x)

    def enable_dropout(self):
        """Enable dropout for MC Dropout uncertainty estimation."""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def disable_dropout(self):
        """Disable dropout for deterministic inference."""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.eval()


class NeuralNetworkModel:
    """
    Neural Network wrapper with training, inference, and persistence.

    This class provides a high-level interface for:
    - Training with validation
    - Making predictions
    - Uncertainty estimation via MC Dropout
    - Model saving and loading
    - Training history tracking
    """

    def __init__(self, config: Optional[NeuralNetConfig] = None):
        """
        Initialize neural network model.

        Args:
            config: Network configuration
        """
        self.config = config or NeuralNetConfig()
        self.model: Optional[FeedforwardNN] = None
        self.device = torch.device(self.config.device)

        # Training state
        self.is_fitted = False
        self.training_history: Dict[str, List[float]] = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': []
        }
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0

        # Feature scaling parameters (mean and std)
        self.feature_mean: Optional[torch.Tensor] = None
        self.feature_std: Optional[torch.Tensor] = None

        logger.info("Initialized NeuralNetworkModel")

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Train the neural network.

        Args:
            X: Training features, shape (n_samples, n_features)
            y: Training targets, shape (n_samples,)
            X_val: Validation features (optional)
            y_val: Validation targets (optional)

        Returns:
            Dict with final training metrics
        """
        logger.info("Starting neural network training")

        # Convert to tensors and normalize
        X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor = \
            self._prepare_data(X, y, X_val, y_val)

        # Create model
        input_dim = X_train_tensor.shape[1]
        self.model = FeedforwardNN(
            input_dim=input_dim,
            hidden_layers=self.config.hidden_layers,
            dropout_rate=self.config.dropout_rate
        ).to(self.device)

        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )

        if X_val_tensor is not None:
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False
            )
        else:
            val_loader = None

        # Setup optimizer and loss
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=self.config.lr_scheduler_factor,
            patience=self.config.lr_scheduler_patience
        )

        criterion = nn.MSELoss()

        # Reset training state
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': []
        }
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0

        # Training loop
        for epoch in range(self.config.epochs):
            # Train
            train_loss, train_mae = self._train_epoch(
                train_loader, optimizer, criterion
            )
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_mae'].append(train_mae)

            # Validate
            if val_loader is not None:
                val_loss, val_mae = self._validate(val_loader, criterion)
                self.training_history['val_loss'].append(val_loss)
                self.training_history['val_mae'].append(val_mae)

                # Learning rate scheduling
                scheduler.step(val_loss)

                # Early stopping
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.epochs_without_improvement = 0
                    # Save best model state
                    self.best_model_state = self.model.state_dict()
                else:
                    self.epochs_without_improvement += 1

                logger.info(
                    f"Epoch {epoch+1}/{self.config.epochs}: "
                    f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                    f"train_mae={train_mae:.4f}, val_mae={val_mae:.4f}"
                )

                # Early stopping check
                if self.epochs_without_improvement >= self.config.early_stopping_patience:
                    logger.info(
                        f"Early stopping triggered at epoch {epoch+1}. "
                        f"Best val_loss: {self.best_val_loss:.4f}"
                    )
                    # Restore best model
                    self.model.load_state_dict(self.best_model_state)
                    break
            else:
                logger.info(
                    f"Epoch {epoch+1}/{self.config.epochs}: "
                    f"train_loss={train_loss:.4f}, train_mae={train_mae:.4f}"
                )

        self.is_fitted = True

        # Calculate final metrics
        metrics = {
            'final_train_loss': self.training_history['train_loss'][-1],
            'final_train_mae': self.training_history['train_mae'][-1],
        }

        if val_loader is not None:
            metrics['final_val_loss'] = self.best_val_loss
            metrics['final_val_mae'] = min(self.training_history['val_mae'])

        logger.info(f"Training completed. Metrics: {metrics}")
        return metrics

    def _prepare_data(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Tuple[torch.Tensor, ...]:
        """
        Prepare and normalize data.

        Returns:
            Tuple of (X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor)
        """
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1).to(self.device)

        # Compute normalization parameters from training data
        self.feature_mean = X_train_tensor.mean(dim=0)
        self.feature_std = X_train_tensor.std(dim=0) + 1e-8  # Avoid division by zero

        # Normalize training data
        X_train_tensor = (X_train_tensor - self.feature_mean) / self.feature_std

        # Normalize validation data if provided
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).reshape(-1, 1).to(self.device)
            X_val_tensor = (X_val_tensor - self.feature_mean) / self.feature_std
        else:
            X_val_tensor = None
            y_val_tensor = None

        return X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor

    def _train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module
    ) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_mae = 0.0
        n_batches = 0

        for X_batch, y_batch in train_loader:
            # Forward pass
            y_pred = self.model(X_batch)
            loss = criterion(y_pred, y_batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track metrics
            total_loss += loss.item()
            total_mae += torch.abs(y_pred - y_batch).mean().item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        avg_mae = total_mae / n_batches

        return avg_loss, avg_mae

    def _validate(
        self,
        val_loader: DataLoader,
        criterion: nn.Module
    ) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        n_batches = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                y_pred = self.model(X_batch)
                loss = criterion(y_pred, y_batch)

                total_loss += loss.item()
                total_mae += torch.abs(y_pred - y_batch).mean().item()
                n_batches += 1

        avg_loss = total_loss / n_batches
        avg_mae = total_mae / n_batches

        return avg_loss, avg_mae

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Input features, shape (n_samples, n_features)

        Returns:
            Predictions, shape (n_samples,)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")

        self.model.eval()

        # Convert and normalize
        X_tensor = torch.FloatTensor(X).to(self.device)
        X_tensor = (X_tensor - self.feature_mean) / self.feature_std

        with torch.no_grad():
            predictions = self.model(X_tensor)

        return predictions.cpu().numpy().flatten()

    def predict_with_uncertainty(
        self,
        X: np.ndarray,
        n_samples: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with uncertainty estimation using MC Dropout.

        Args:
            X: Input features
            n_samples: Number of forward passes with dropout

        Returns:
            Tuple of (mean_predictions, uncertainties)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")

        # Enable dropout
        self.model.enable_dropout()

        # Convert and normalize
        X_tensor = torch.FloatTensor(X).to(self.device)
        X_tensor = (X_tensor - self.feature_mean) / self.feature_std

        # Multiple forward passes
        predictions = []
        for _ in range(n_samples):
            with torch.no_grad():
                pred = self.model(X_tensor)
            predictions.append(pred.cpu().numpy().flatten())

        predictions = np.array(predictions)

        # Compute statistics
        mean_pred = np.mean(predictions, axis=0)
        uncertainty = np.std(predictions, axis=0)

        # Restore normal mode
        self.model.disable_dropout()

        return mean_pred, uncertainty

    def save(self, filepath: str):
        """
        Save model to disk.

        Args:
            filepath: Directory path to save model
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted model")

        filepath = Path(filepath)
        filepath.mkdir(parents=True, exist_ok=True)

        # Save model weights
        torch.save(self.model.state_dict(), filepath / 'model_weights.pt')

        # Save normalization parameters
        torch.save({
            'feature_mean': self.feature_mean,
            'feature_std': self.feature_std
        }, filepath / 'normalization.pt')

        # Save config and history
        metadata = {
            'config': {
                'hidden_layers': self.config.hidden_layers,
                'dropout_rate': self.config.dropout_rate,
                'learning_rate': self.config.learning_rate,
                'batch_size': self.config.batch_size,
                'weight_decay': self.config.weight_decay,
                'device': self.config.device
            },
            'training_history': self.training_history,
            'input_dim': self.model.input_dim
        }

        with open(filepath / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Model saved to {filepath}")

    def load(self, filepath: str):
        """
        Load model from disk.

        Args:
            filepath: Directory path containing saved model
        """
        filepath = Path(filepath)

        # Load metadata
        with open(filepath / 'metadata.json', 'r') as f:
            metadata = json.load(f)

        # Reconstruct model
        input_dim = metadata['input_dim']
        hidden_layers = metadata['config']['hidden_layers']
        dropout_rate = metadata['config']['dropout_rate']

        self.model = FeedforwardNN(
            input_dim=input_dim,
            hidden_layers=hidden_layers,
            dropout_rate=dropout_rate
        ).to(self.device)

        # Load weights
        self.model.load_state_dict(
            torch.load(filepath / 'model_weights.pt', map_location=self.device)
        )

        # Load normalization parameters
        norm_params = torch.load(filepath / 'normalization.pt', map_location=self.device)
        self.feature_mean = norm_params['feature_mean']
        self.feature_std = norm_params['feature_std']

        # Load history
        self.training_history = metadata['training_history']
        self.is_fitted = True

        logger.info(f"Model loaded from {filepath}")


__all__ = [
    'NeuralNetConfig',
    'FeedforwardNN',
    'NeuralNetworkModel'
]
