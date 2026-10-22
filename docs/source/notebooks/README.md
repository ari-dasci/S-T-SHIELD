# Notebooks

`model.pth` is **not distributed with this repository**. It is a checkpoint produced by
`train.ipynb`, which ends with `torch.save(classifier.state_dict(), ...)` writing
`./model.pth` next to the notebook; `test_REVEL.ipynb` then loads it.

Run `train.ipynb` first, then `test_REVEL.ipynb`.

The checkpoint used to be committed here. It was removed from the working tree and from the
git history because it weighed more than the rest of the project combined, and it is
reproducible by running the training notebook.
