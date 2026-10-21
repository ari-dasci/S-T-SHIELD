# T-SHIELD

[![PyPI](https://img.shields.io/pypi/v/tshield-xai.svg)](https://pypi.org/project/tshield-xai/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

> A regularization family that hides part of the input during training, so the model cannot
> lean on a handful of dominant features — improving both its performance and the
> explanations it produces.

T-SHIELD (Transformation-Selective Hidden Input Evaluation for Learning Dynamics) adds a
penalty term to the training objective that measures how much the prediction depends on a
small subset of the input. Because the penalty is computed from the model itself, no
human-labelled explanations are needed: the model is nudged towards using more of the
evidence available.

## Two variants

| Function | How it picks the region to hide |
| --- | --- |
| `rshield` (R-SHIELD) | Random regions of the input. |
| `xshield` (X-SHIELD) | Regions the model itself considers least important, taken from its own explanation. |

Both share the same signature and return a scalar penalty to be added to the loss:

```python
rshield(model, input, input_0=None, segmentation=1, device="cpu", percentage=None)
xshield(model, input, input_0=None, segmentation=1, device="cpu", percentage=None)
```

## Installation

The package is published on PyPI as `tshield-xai` (Python 3.8+):

```bash
pip install tshield-xai
```

From source:

```bash
git clone https://github.com/ari-dasci/S-T-SHIELD.git
cd S-T-SHIELD
pip install .
```

## Quick start

The regularizer plugs into a normal training loop as an extra term. `procedures.train_step`
already accepts it:

```python
import torch
from TSHIELD.procedures import procedures
from TSHIELD.TSHIELD import xshield

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loss_f = torch.nn.CrossEntropyLoss()
reg_f = lambda x, y: xshield(model=x, input=y, percentage=0.3, device=device)

# One full epoch: the returned `reg` is the regularization term for that epoch.
loss, acc, reg = procedures.train_step(
    train_loader, model, optimizer, loss_f, reg_f, device, transform=None, train=True
)
print(f"loss={loss:.4f}  acc={acc:.2f}  reg={reg:.4f}")
```

Swap `xshield` for `rshield` to train with the random-region variant instead. A complete
training and validation script, including the argument parser and the metrics used in the
paper, lives in [`TSHIELD/tests/train.py`](TSHIELD/tests/train.py).

Set `percentage` to control how much of the input is hidden; leaving it `None` uses the
function's own default.

## Documentation

API documentation is at <https://s-t-shield.readthedocs.io/en/latest/>.

## Paper and citation

The method was published as:

> Sevillano-García, I., Luengo, J., Herrera, F. (2026).
> *X-SHIELD: Regularization for explainable Artificial Intelligence.*
> Machine Intelligence Research, 23. <https://doi.org/10.1007/s11633-025-1576-y>

An earlier preprint is also available as arXiv:2404.02611. If you use this code, please
cite the published version:

```bibtex
@article{sevillano2026xshield,
  title   = {X-SHIELD: Regularization for explainable Artificial Intelligence},
  author  = {Sevillano-Garc{\'i}a, Iv{\'a}n and Luengo, Juli{\'a}n and Herrera, Francisco},
  journal = {Machine Intelligence Research},
  volume  = {23},
  year    = {2026},
  doi     = {10.1007/s11633-025-1576-y}
}
```

```bibtex
@article{sevillano2024shield,
  title   = {SHIELD: A regularization technique for eXplainable Artificial Intelligence},
  author  = {Sevillano-Garc{\'i}a, Iv{\'a}n and Luengo, Juli{\'a}n and Herrera, Francisco},
  journal = {arXiv preprint arXiv:2404.02611},
  year    = {2024}
}
```

## License

GNU General Public License v3.0 — see the [LICENSE](LICENSE) file.

## Authors

- **Iván Sevillano-García** — [isevillano@ugr.es](mailto:isevillano@ugr.es)
- Julián Luengo
- Francisco Herrera

Research Group: [Andalusian Research Institute in Data Science and Computational
Intelligence (DaSCI)](https://dasci.es), University of Granada, Spain.
