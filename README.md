# abiospath_lyu
# Predicting the Risk of Ischemic Stroke in Patients with Atrial Fibrillation using Heterogeneous Drug-Protein-Disease Network-Based Deep Learning

ABioSPATH is a deep learning model designed to predict the risk of ischemic stroke in patients with atrial fibrillation. It uses a heterogeneous drug-protein-disease network to understand and model the complex interactions that lead to this condition.

## Repository Structure

This repository contains the following:

- `.idea/` : Contains project settings and configurations for JetBrains IDEs.
- `base/` : Base modules that are used across the project.
- `config/` : Contains configuration files for the project.
- `dataloader/` : Scripts for loading and preprocessing data.
- `logger/` : Logging utilities for tracking the project's progress.
- `trainer/` : Training scripts for the model.
- `utils/` : Utility scripts for miscellaneous tasks.
- `model.py` : The script that defines the deep learning model.
- `testfile.py` : Script to test the trained model.
- `trainandvalid.py` : Main script to train and validate the model.

Files like `.DS_Store` and `README.md` are used for repository management and documentation.

## Running the Project

The main scripts to run are `trainandvalid.py` for training and validating the model and `testfile.py` for testing the model's performance.

You can modify the default parameters for the scripts in the `config/` directory. The current parameters are stored in `config/default.json` (for example).

To train the model, run:

```bash
python trainandvalid.py
```

to test the model, run:
```bash
python test.py
```



## Running the Project
Data used in this project is available upon request. Please send an email to [zhiheng.lyu@my.cityu.edu]

