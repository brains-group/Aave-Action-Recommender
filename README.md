# Aave Action Recommender

This repository contains the Aave Action Recommender system, which depends on the Aave Simulator for its functionality.

## Git Submodule

This repository uses **Aave-Simulator** and **Aave-Data-Pipeline** as Git submodules.

Initialize both `Aave-Simulator` and `Aave-Data-Pipeline` after cloning with `git -c protocol.file.allow=always submodule update --init --recursive`. The local pipeline submodule URL must be available on the machine or replaced with an authorized remote.

## Setup

`conda create --name aave-action-recommender python=3.12.4`

`conda activate aave-action-recommender`

`pip install -r requirements.txt`

## Attribution

If you use the code in this repository, please cite the following corresponding paper:
```
@inproceedings{spadea2026from,
  author = {Spadea, Fernando and Seneviratne, Oshani},
  title = {{From Risk to Rescue: An Agentic Survival Analysis Framework for Liquidation Prevention}},
  year = {2026},
  booktitle = {{IEEE International Conference on Blockchain and Cryptocurrency}},
}
```
