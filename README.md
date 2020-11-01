# off-policy-mlsh
## Installation
1. Install tianshou `git clone https://github.com/thu-ml/tianshou.git`
Cd to the directory tianshou root directory and delete gym dependency in tianshou setup.py. Then `pip install -e .` 
2. run `pip install -r requirements.txt`
3. Add to your `.bash_profile` (replace ... with path to directory):
    ```
    export PYTHONPATH=$PYTHONPATH:/.../mlsh-proj/gym;
    export PYTHONPATH=$PYTHONPATH:/.../mlsh-proj/rl-algs;
    ```
4. `cd` into `gym` then run `pip install -e .` inside that directory.
5. `cd` into `test_envs` then run `pip install -e .` inside that directory.

## Dependices

- pytorch 1.5.0
- wandb 0.8.36
- mujoco-py 0.5.7
- MuJoCo 1.3.1
- Python3.6
- pyglet 1.3.1

## Running experiments

cd into `mlsh_dqn` and run
```
python train.py -W 60 -U 1 -T 50 --env MovementBandits-v0
```
Change `-W` to adjust warm-up period length, `-U` to adjust joint update period, and `-T` for lengh of each rollout episode. Use `--env` to specify the environment to run experiments on, `AntBandits-v1` and `MovementBandits-v0` are supported. More options and usages can be find by running `python train.py -h`