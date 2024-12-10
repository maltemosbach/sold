# SOLD: Slot-Attention for Object-centric Latent Dynamics

**[AIS, University of Bonn](https://www.ais.uni-bonn.de/index.html)**

[Malte Mosbach](https://maltemosbach.github.io/)&ast;, [Jan Niklas Ewertz]()&ast;, [Angel Villar-Corrales](http://angelvillarcorrales.com/templates/home.php), [Sven Behnke](https://www.ais.uni-bonn.de/behnke/)

[[`Paper`](https://arxiv.org/abs/2410.08822)] &nbsp; [[`Website`](https://slot-latent-dynamics.github.io/)] &nbsp; [[`BibTeX`](https://slot-latent-dynamics.github.io/bibtex.txt)]

**Slot-Attention for Object-centric Latent Dynamics (SOLD)** is a model-based reinforcement learning algorithm operating on a structured latent representation in its world model.

![SOLD Overview](assets/sold_overview.png)


[//]: # (<img src="docs/sample_rollout.png" width="100%"><br/>)

## Getting started
Begin by installing the [multi-object-fetch](https://github.com/maltemosbach/multi-object-fetch) environment suite.
Thereafter, in the same conda environment, install SOLD and its dependencies:
```bash
pip install -e .
```


## Training
SAVi models are pretrained on static datasets of random trajectories. Such datasets can be generated using the `generate.py`[./src/sold/dataa] script.
First to pre-train a SAVi model, run:
```bash
python train_savi.py experiment=my_savi_model
```

<img src="assets/savi_reach_red.png" width="40%"> &nbsp; <img src="assets/savi_pick_red.png" width="40%">

To train SOLD, a checkpoint path to the pre-trained SAVi model is required, which can be specified in the [`train_sold.yaml`](./src/sold/configs/train_sold.yaml) configuration file.
Then, to start the training, run:
```bash
python train_sold.py
```


All results are stored in the [`experiments`](./experiments) directory.
To further evaluate a trained model or a set of models in a directory, you can run 
```bash
python eval_sold.py checkpoint_path=PATH_TO_CHECKPOINT(S)
```
which will create metrics and visualizations for the checkpoints.

## Checkpoints
We added pre-trained SAVi and SOLD models in the [`checkpoints`](./checkpoints) directory.

