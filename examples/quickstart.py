"""Runnable end-to-end example of T-SHIELD.

No dataset and no download: a small CNN with random weights, a random batch, and one
optimiser step per variant. The point is to show where the regularizer goes in a training
loop, not to produce meaningful accuracies.

Run from the repository root:

    python examples/quickstart.py
"""

import torch
import torch.nn as nn

from TSHIELD.TSHIELD import rshield, xshield

torch.manual_seed(0)

IMG_SIZE = 32
N_CLASSES = 4
BATCH = 4
PERCENTAGE = 30  # percentage of the input to hide, 0-100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TinyNet(nn.Module):
    """Stand-in for the model being regularized."""

    def __init__(self, n_classes: int = N_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(16, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x).flatten(1))


images = torch.rand(BATCH, 3, IMG_SIZE, IMG_SIZE)
# The regularizers differentiate the model output with respect to the input, so the batch
# must be part of the autograd graph.
images.requires_grad_(True)
labels = torch.randint(0, N_CLASSES, (BATCH,))
criterion = nn.CrossEntropyLoss()


def train_step(model, reg_f=None):
    """One optimisation step. `reg_f` receives the batch and returns a scalar penalty."""
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.zero_grad()

    loss = criterion(model(images), labels)
    reg = reg_f(images) if reg_f is not None else torch.zeros(())
    (loss + reg).backward()
    optimizer.step()

    return loss.item(), (reg.detach().item() if torch.is_tensor(reg) else float(reg))


def fresh_model():
    return TinyNet().to(device)


# The penalty is printed in scientific notation: on an untrained model the softmax outputs
# are nearly uniform, so occlusion barely moves them and the term is tiny. It grows as the
# model becomes confident during real training.
model = fresh_model()
loss, reg = train_step(model)
print(f"baseline        loss={loss:.4f}  reg={reg:.3e}")

model = fresh_model()
loss, reg = train_step(
    model,
    reg_f=lambda x: rshield(model=model, input=x, percentage=PERCENTAGE, device=device),
)
print(f"R-SHIELD        loss={loss:.4f}  reg={reg:.3e}")

model = fresh_model()
loss, reg = train_step(
    model,
    reg_f=lambda x: xshield(model=model, input=x, percentage=PERCENTAGE, device=device),
)
print(f"X-SHIELD        loss={loss:.4f}  reg={reg:.3e}")

print(
    "\nThe baseline has no penalty, so its reg is zero by construction. The three runs start "
    "from the same weights, so the losses differ only because of the extra term.\n"
    "The numbers are not meaningful: the model is randomly initialised on purpose."
)
