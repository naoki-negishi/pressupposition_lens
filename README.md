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

### 1. Build docker image and set up the environment
- Before building a docker image, you need to write your username to the `.env` file.
  ```bash
  cd docker

  # write your username to .env file
  echo "HOME=$HOME" > .env
  echo "HOST_USERNAME=$(id -un)" >> .env
  ```
- You can build a docker image #TODO: add
  ```bash
  docker compose up --build -d

  docker exec -it pres_lens /bin/zsh
  ```
#TODO: docker-compose.yml personal directories

- After building a docker image, you can set up the environment by running the following command.
  ```bash
  sh setting.sh
  ```

### 2. Prepare a dataset of premise-hypothesis pairs
- We recommend the following three datasets:
  1. IMPPRES dataset ([paper](https://aclanthology.org/2020.acl-main.768/))
  2. PROPRES dataset ([paper](https://aclanthology.org/2023.conll-1.9/))
  3. NOPE dataset ([paper](https://aclanthology.org/2021.conll-1.28/))

### 3. Train an Affine convertion, and evaluate it
- You can create an Affine convertion by running the following command.
  ```bash
  CUDA_VISIBLE_DEVICES=0,1,2,3 poetry run python run.py --yaml_file <path_to_config_yaml>
  ```

- In the outputs directory, you can see the following files:
  - `excution_{time}.log`: a log file of evaluation
  - `lre_weight.pth`: LRE weight tensor
  - `lre_bais.pth`: LRE bias tensor
  - `visualized_img.png`: visualized image
