# Deep Learning Knowledge Tracing

This repository contains code for deep learning knowledge tracing, a technique used in educational technology to model and predict student learning. Knowledge tracing algorithms aim to estimate a student's mastery level of a particular concept or skill over time based on their interaction with learning materials.

## Overview

The main script `train.py` in this repository orchestrates the process of training deep learning models for knowledge tracing. It integrates various modules for data loading, model selection, optimization, and training.

## Usage

To run the knowledge tracing process, follow these steps:

1. Install the required dependencies listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

2. Configure the training settings and hyperparameters by modifying the `define_argparser.py` file.

3. Execute the main script `train.py` with the following command:

```bash
python train.py
```

## File Structure

- `train.py`: Main script for orchestrating the knowledge tracing process.
- `get_modules/`: Directory containing modules for data loading, model selection, and training.
- `utils.py`: Utility functions for optimization, criterion selection, recording, and visualization.
- `define_argparser.py`: Configuration file for specifying training settings and hyperparameters.
- `model_records/`: Directory for storing trained model checkpoints and logs.

## Dependencies

The following Python packages are required to run the code:

- numpy
- torch
- scikit-learn

Install the dependencies using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## References

    Unggi Lee's github page

https://github.com/codingchild2424/Deep_knowledge_tracing_baseline    

    Juno-hwang's Juno-dkt github page

https://github.com/juno-hwang/juno-dkt

    Hyungcheol Noh's blog and github page

https://github.com/hcnoh/knowledge-tracing-collection-pytorch

https://hcnoh.github.io/2019-06-14-deep-knowledge-tracing

    Kim, Ki Hyun's lectures in Fastcampus

https://fastcampus.co.kr/

    Deep Knowledge Tracing Paper

Piech, C., Spencer, J., Huang, J., Ganguli, S., Sahami, M., Guibas, L., & Sohl-Dickstein, J. (2015). Deep knowledge tracing. arXiv preprint arXiv:1506.05908.