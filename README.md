# Reinforcement Learning Communication
learned communication in multiagent reinforcement learning

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
pip install -re ~/path/to/rl_comm/requirements.txt
```

Suggestions to improve this documentation or workflow are welcomed.
