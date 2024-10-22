# SOLD: Slot-Attention for Latent Object-centric Dynamics

Slot-Attention for Object-centric Latent Dynamics is a model-based reinforcement learning algorithm operating on a structured latent representation in its world model.

## Getting started
Install via conda or pip ...


## Training

First to pre-train a SAVi model, run:
```bash
python training/savi.py experiment=my_exp
```

Then, to train the SOLD model, run:
```bash
python training/sold.py
```

The results are stored in the [`experiments`](./experiments) directory.