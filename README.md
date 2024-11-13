# SOLD: Slot-Attention for Object-centric Latent Dynamics

**[AIS, University of Bonn](https://www.ais.uni-bonn.de/index.html)**

[Malte Mosbach](https://maltemosbach.github.io/), [Jan Niklas Ewertz](), [Angel Villar-Corrales](http://angelvillarcorrales.com/templates/home.php), and [Sven Behnke](https://www.ais.uni-bonn.de/behnke/)

[[`Paper`](https://arxiv.org/abs/2410.08822)] &nbsp; [[`Website`](https://slot-latent-dynamics.github.io/)] &nbsp; [[`BibTeX`](https://slot-latent-dynamics.github.io/bibtex.txt)]

Slot-Attention for Object-centric Latent Dynamics (SOLD) is a model-based reinforcement learning algorithm operating on a structured latent representation in its world model.

![SOLD Overview](assets/sold_overview.png)


[//]: # (<img src="docs/sample_rollout.png" width="100%"><br/>)

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


## Checkpoints
We added pre-trained SAVi and SOLD models in the [`checkpoints`](./checkpoints) directory.




## Structure

```
┌── sold
│   ├── algorithms
│   │   ├── savi.py : Training-loop for SAVi encoder-decoder model.
│   │   └── sold.py : Training-loop for SOLD based on a pre-trained SAVi model.
│   ├── datasets
│   │   ├── experience_source.py : Dataset for SOLD that samples sequences from the replay buffer.
│   │   └── image_folder.py : Load dataset for SAVi from image folder.
│   ├── envs
│   │   ├── image_env.py : Defines the visual environment interface used by SOLD.
│   │   └── wrappers.py : ...
│   ├── models
│   │   ├── savi
│   │       ├── corrector.py : Defines the visual environment interface used by SOLD.
│   │       └── decoder.py : ...
│   │   └── sold
│   │         └ input : deterministic and stochastic and action
│   │         └ output : embedded observation
│   └── utils
│       ├── buffer.py : Contains the replay buffer used to store and sample transitions during training
│       └── utils.py : Contains other utility functions
└── main.py : Reads the configuration file, sets up the environment, and starts the training process
```