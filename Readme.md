# Toy Tabular Experiments for Tiered RL Setting
Code for General Tiered RL paper https://arxiv.org/abs/2302.05534. If you find it helpful, please cite as follow:

```
@article{huang2023robust,
  title={Robust Knowledge Transfer in Tiered Reinforcement Learning},
  author={Huang, Jiawei and He, Niao},
  journal={arXiv preprint arXiv:2302.05534},
  year={2023}
}
```

## How to run experiments


### Run experiments with different number of source tasks:
python3 Algorithm.py -W 1 --model-seed 1000 --delay-transfer 500000 --lam 0.3 --seed 100 200 300 400 500 ...

python3 Algorithm.py -W 2 --model-seed 1000 --delay-transfer 500000 --lam 0.3 --seed 100 200 300 400 500 ...

python3 Algorithm.py -W 3 --model-seed 1000 --delay-transfer 500000 --lam 0.3 --seed 100 200 300 400 500 ...

python3 Algorithm.py -W 5 --model-seed 1000 --delay-transfer 500000 --lam 0.3 --seed 100 200 300 400 500 ...


### Plot results
python3 draw.py --log-dir [Your Exp. Log Dirs] -W 0 1 2 5