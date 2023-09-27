# Careful at Estimation and Bold at Exploration for Deterministic Policy Gradient Algorithm

## Virtualenv Configuration
please refer to the "requirements.txt" file in the "configure" folder. We have listed some key packages in that file. By installing these packages, their dependencies will also be automatically installed.

## Hardware
we conducted them on a desktop computer with an Intel i9-9900KF CPU, Nvidia RTX 2070 Super GPU, and 64GB of RAM.

## How to run BACC algo
After setting up the virtual environment, you can use the following command to run the program once.
```shell
python3 -um main --algo=bacc --seed=1 --domain=humanoid --task=test --sample_range=5 --sample_size=32 --beta=0.01
```
