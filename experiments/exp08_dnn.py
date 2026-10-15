from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline import (
    build_preprocessor,
    cat_features,
    num_features,
)


BATCH_SIZE = 32
WEIGHT_DECAY = 0.01
EPOCHS = 75
LEARNING_RATE = 0.001
HIDDEN_SIZE = 32
DROPOUT_RATE = 0.1
MODEL_RANDOM_STATE = 42
CV_RANDOM_STATE = 42

np.random.seed(MODEL_RANDOM_STATE)
torch.manual_seed(MODEL_RANDOM_STATE)

features = num_features + cat_features

train_data = pd.read_csv(
    PROJECT_ROOT / "titanic" / "train.csv"
)

X = train_data[features].copy()
y = train_data["Survived"].to_numpy()

CV = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=CV_RANDOM_STATE,
)

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cpu"
)

class TitanicDNN(nn.Module):
    """Small feed-forward network used in the DNN experiments."""

    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(HIDDEN_SIZE, 1),
        )

    def forward(self, x):
        return self.network(x)
            
fold_scores = []

for fold, (train_idx, val_idx) in enumerate(
    CV.split(X, y),
    start=1,
):
    preprocessor = build_preprocessor(
        scale_numeric=True,
        dense_output=True,
    )

    X_train = preprocessor.fit_transform(
        X.iloc[train_idx]
    )
    X_val = preprocessor.transform(
        X.iloc[val_idx]
    )

    y_train = y[train_idx]
    y_val = y[val_idx]

    print(
        f"Fold {fold}: "
        f"train={X_train.shape}, "
        f"validation={X_val.shape}"
    )
    
    X_train_tensor = torch.tensor(
        X_train,
        dtype=torch.float32,
    )
    X_val_tensor = torch.tensor(
        X_val,
        dtype=torch.float32,
    )

    y_train_tensor = torch.tensor(
        y_train,
        dtype=torch.float32,
    ).reshape(-1, 1)
        
    y_val_tensor = torch.tensor(
        y_val,
        dtype=torch.float32,
    ).reshape(-1, 1)

    train_dataset = TensorDataset(
        X_train_tensor,
        y_train_tensor,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    model = TitanicDNN(
        input_size=X_train_tensor.shape[1]
    ).to(device)

    loss_function = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    for _ in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            logits = model(batch_X)
            loss = loss_function(logits, batch_y)

            loss.backward()
            optimizer.step()

            total_loss += (
                loss.item() * batch_X.size(0)
            )

    mean_train_loss = (
        total_loss / len(train_dataset)
    )

    model.eval()

    with torch.no_grad():
        val_logits = model(
            X_val_tensor.to(device)
        )
        val_loss = loss_function(
            val_logits,
            y_val_tensor.to(device),
        ).item()

        val_probabilities = torch.sigmoid(
            val_logits
        )

        val_predictions = (
            val_probabilities >= 0.5
        ).to(torch.int32)

    val_predictions = (
        val_predictions
        .cpu()
        .numpy()
        .reshape(-1)
    )

    fold_accuracy = accuracy_score(
        y_val,
        val_predictions,
    )

    fold_scores.append(fold_accuracy)

    print(
        f"Fold {fold}: "
        f"train_loss={mean_train_loss:.4f}, "
        f"val_loss={val_loss:.4f}, "
        f"accuracy={fold_accuracy:.4f}"
    )

fold_scores = np.array(fold_scores)

print(
    f"Mean CV accuracy: {fold_scores.mean():.4f}"
)

print(
    f"CV std: {fold_scores.std():.4f}"
)
