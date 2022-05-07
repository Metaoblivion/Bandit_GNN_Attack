# Bandit_GNN_Attack

This repo contains the code, data and results reported in our paper.

If you make use of the code/experiment in your work, please cite our paper (Bibtex below).

```
@inproceedings{wang2022bandits,
title={Bandits for Structure Perturbation-based Black-box Attacks to Graph Neural Networks with Theoretical Guarantees},
author={Wang, Binghui and Li, Youqi and Zhou, Pan},
booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
year={2022}
}
```

# Requirement

Pytorch
dgl
ogb
numpy
scipy

# Run

We have provide the configs for GCN, SGC, GIN for the corresponding datasets.

Taking GCN for citeseer as an example, the command to run our code is

python blackbox.py -c config/config_GCN_citeseer.json

# Extension

Our code can be extended to handle other models and datasets.

1. New a .py file in models directory to define the model.
2. Place the model parameter file in modeldata directory.
3. Place the dataset in data directory.
4. Determine the target set you aim to attack and place the serialized file in attackSet.
5. New a config file in config directory.
