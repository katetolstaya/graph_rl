# Reinforcement Learning Communication
Learned communication in multiagent reinforcement learning.

<img src="https://user-images.githubusercontent.com/2522557/65058418-36aeb900-d942-11e9-8ad4-bb34ffd9eb1a.jpg" width="300">

# Recommended Installation

Create and activate a new virtual environment, ([venv docs](https://docs.python.org/3/tutorial/venv.html)) then upgrade pip for the venv.


```
python3 -m venv ~/venv/my_rl_venv
source ~/venv/my_rl_venv/bin/activate
pip install --upgrade pip
```

Install TensorFlow 1.x; choose either with or without GPU support.

```
if gpu,    pip install tensorflow-gpu
if no gpu, pip install tensorflow
```

Install StableBaselines prerequisites (MPI not required).

https://github.com/hill-a/stable-baselines

Clone the environment engine, gym environment, and this repo.

```
git clone https://github.com/jpaulos/perimeter_defense.git
git clone https://github.com/jpaulos/gym_pdefense.git
git clone https://github.com/jpaulos/rl_comm.git
```

Install from source the environment engine, gym environment, and this repo.

```
pip install -e ~/path/to/gym_pdefense
pip install -e ~/path/to/perimeter_defense
pip install -e ~/path/to/rl_comm
```

Optional. The previous installations should automatically include the minimum dependencies for each of these packages. The additional packages listed in requirements.txt will provide a more complete development environment ([requirements.txt docs](https://packaging.python.org/discussions/install-requires-vs-requirements/)).

```
pip install -r ~/path/to/rl_comm/requirements.txt
```

# Demo Testing with Pre-Trained Model

Download the pretrained checkpoint file and save in the next to demo_test.py.

https://www.dropbox.com/s/mdqjbi79kc9cwj8/ckpt_050.pkl?dl=0

Run demo_test.py to see the final results of 100 games, followed by live visualizations of games.

```
python demo_test.py
```

This is a brittle demo requiring the checkpoint file and current version of the model code to be somewhat in sync, and is likely to be broken before the code base stabalizes. A more consistent workflow for organizing and versioning trained models is needed.

# Info

The team policy is built using graph network primitives from DeepMind's [Graph Nets](https://github.com/deepmind/graph_nets) library, which is built on top of Tensorflow and [Sonnet](https://github.com/deepmind/sonnet/tree/master).

# Contributing

Suggestions to improve this documentation or workflow are welcomed.
