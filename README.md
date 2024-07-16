# Presupposition Lens (tentative)

## Overview
- This is a tool to convert a premise sentence, which contains a presupposition trigger, to a hypothesis sentence.
- Especially, following to the previous work, you can get convertation method from only some examples.
  - See [the previous work](https://arxiv.org/abs/2308.09124).


## Usage
- You can use this tool by following the steps below.
  1. Build docker image.
  2. Prepare a dataset of premise-hypothesis pairs.
  3. Train a model to convert a premise sentence to a hypothesis sentence.

### 1. Build docker image
- You can build a docker image by running the following command.
  ```bash
  docker build -t presupposition-lens .
  ```
- after this, all the following commands should be run in the docker container.
  ```bash
  docker run -it presupposition-lens
  ```

### 2. Prepare a dataset of premise-hypothesis pairs
- We recommend the following three datasets:
  1. IMPPRES dataset ([paper]())
  2. PROPRES dataset ([paper](https://aclanthology.org/2023.conll-1.9/))
  3. NOPE dataset ([paper]())

### 3. Create an Affine convertion, and evaluate it
- You can create an Affine convertion by running the following command.
  ```bash
  python run.py --yaml_file <path_to_config_yaml>
  ```

- In the outputs directory, you can see the following files:
  - `excution_{time}.log`: a log file of evaluation
  - `lre_weight.pth`: LRE weight tensor
  - `lre_bais.pth`: LRE bias tensor
  - `visualized_img.png`: visualized image
