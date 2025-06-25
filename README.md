# SOLD: Slot-Attention for Object-centric Latent Dynamics

**[AIS, University of Bonn](https://www.ais.uni-bonn.de/index.html)**

[Malte Mosbach](https://maltemosbach.github.io/)&ast;, [Jan Niklas Ewertz]()&ast;, [Angel Villar-Corrales](http://angelvillarcorrales.com/templates/home.php), [Sven Behnke](https://www.ais.uni-bonn.de/behnke/)

[[`Paper`](https://arxiv.org/abs/2410.08822)] &nbsp; [[`Website`](https://slot-latent-dynamics.github.io/)] &nbsp; [[`BibTeX`](https://slot-latent-dynamics.github.io/bibtex.txt)]

**Slot-Attention for Object-centric Latent Dynamics (SOLD)** is a model-based reinforcement learning algorithm operating on a structured latent representation in its world model.

![SOLD Overview](assets/sold_overview.png)


[//]: # (<img src="docs/sample_rollout.png" width="100%"><br/>)

## Installation
### Conda
Start by installing the [multi-object-fetch](https://github.com/maltemosbach/multi-object-fetch) environment suite.
Then add the SOLD dependencies to the conda environment by running:
```bash
conda env update -n mof -f apptainer/environment.yml
```

### Apptainer
Alternatively, we provide an [`Apptainer`](apptainer/multi_object_fetch.def) build file to simplify installation.
To build the `.sif` image, run:
```bash
cd apptainer && apptainer build sold.sif multi_object_fetch.def
```
To start a training run inside the container:
```bash
apptainer run --nv ../sold.sif python train_sold.py
```
> [!NOTE] 
>If you're on a SLURM cluster, you can submit training jobs using this container with the provided run script `sbatch slurm.sh train_sold.py`.



## Training
The training routine consists of two stages: [pre-training a SAVi model](#pre-training-a-savi-model) and 
[training a SOLD model](#training-a-sold-model) on top of it.

### Pre-training a SAVi model
The SAVi models (or autoencoders generally) are pre-trained on static datasets of random trajectories. 
Such datasets can be generated using the following script:
```bash
python generate_dataset.py experiment=my_dataset env.name=ReachRed_0to4Distractors_Dense-v1
```

To train a SAVi model, specify the dataset to be trained on and model parameters such as the number of slots in [`train_autoencoder.yaml`](./configs/train_autoencoder.yaml) and run:
```bash
python train_savi.py experiment=my_savi_model
```

<details>
    <summary><i>Show sample pre-training results</i></summary>
    Good SAVi models should learn to split the scene into meaningful objects and keep slots assigned to the same object over time.
    Examples of SAVi models pre-trained for a reaching and picking task are shown below.
    <img src="assets/savi_reach_red.png" width="49%" align="top"> <img src="assets/savi_pick_red.png" width="49%" align="top">
</details>

### Training a SOLD model

To train SOLD, a checkpoint path to the pre-trained SAVi model is required, which can be specified in the [`train_sold.yaml`](./configs/train_sold.yaml) configuration file.
Then, to start the training, run:
```bash
python train_sold.py
```
All results are stored in the [`experiments`](./experiments) directory.


<details>
    <summary><i>Show sample training outputs</i></summary>
    When training a SOLD model, you can check different visualisations to monitor the training progress. 
    The <i>dynamics_prediction</i> plot highlights the differences between the ground truth and the predicted future states, and 
    shows the forward prediction of each slot.
    <p align="center">
      <img src="assets/dynamics_reach_red.png" width="100%">
    </p>
    In addition, visualisations of <i>actor_attention</i> or <i>reward_predictor_attention</i>, as shown below, can be used to 
    understand what the model is paying attention to when predicting the current reward, i.e. which elements of the scene 
    the model considers to be reward-predictive.
    <p align="center">
      <img src="assets/reward_predictor_attention_reach_red.png" width="100%">
    </p>
</details>



For further evaluation of a trained model or a set of models in a directory, you can run:
```bash
python evaluate_sold.py checkpoint_path=PATH_TO_CHECKPOINT(S)
```
which will log performance metrics and visualizations for the given checkpoints.

## Checkpoints
Pre-trained SAVi and SOLD models are available in the [`checkpoints`](./checkpoints) directory.
The SAVi checkpoints can be used to begin training SOLD models right away.
Each checkpoint also includes corresponding TensorBoard logs, allowing you to visualize the expected training dynamics:
```bash
tensorboard --logdir checkpoints
```


## Citation
If you find this work useful, please consider citing our paper as follows:
```bibtex
@inproceedings{sold2025mosbach,
  title={SOLD: Slot Object-Centric Latent Dynamics Models for Relational Manipulation Learning from Pixels},
  author={Malte Mosbach and Jan Niklas Ewertz and Angel Villar-Corrales and Sven Behnke},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2025}
}
```
