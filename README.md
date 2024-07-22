# DeePref: Deep Reinforcement Learning For Video Prefetching In Content Delivery Networks

DeePref is a deep reinforcement learning agent for prefetching video content on edge networks in Content Delivery Networks (CDNs).
DeePref leverages [DQN](https://www.nature.com/articles/nature14236) and [DRQN](https://arxiv.org/abs/1507.06527) to effectively encode the agent's history to make auto-aggressive prefetching decisions at edge networks.
DeePref is a prefetcher that is agnostic to the hardware design, operating system, and applications/workloads in which it utilizes only the video ID to effectively make future prefetching decisions. DeePref outperforms baseline approaches that use video
content popularity as a building block to statically or dynamically make prefetching decision.

Our paper is available on arXiv! You can read it [here](https://arxiv.org/abs/2310.07881).


## Table of Contents
- [Deep Reinforcement Learning For Video Prefetching In Content Delivery Networks](#deep-reinforcement-learning-for-video-prefetching-in-content-delivery-networks)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
    - [Creating Environment](#creating-environment)
    - [Installing Pytorch](#installing-pytorch)
  - [Running Experiments](#running-experiments)
    - [Experiment Argument Details](#experiment-argument-details)
  - [Citing DeePref](#citing-dtqn)
  - [Contributing](#contributing)

## Installation

To run our code, you must first set up your environment.
We recommend using `anaconda` with `pip` to set up a virtual environment and dependency manager.
For our experiments, we use `python3.10`; while other versions of python will probably work, they are untested and we cannot guarantee the same performance.

### Creating Environment

First, create a conda environement with `python3.10`, then install the required dependencies. This can be done with:

```bash
conda create -n DeePref python=3.10
conda activate DeePref
pip install -r requirements.txt
```

### Installing Pytorch

Second, install 'pytorch' into your virtual environement from [source](https://pytorch.org/get-started/locally/). For our experiments, we used 'pytorch' version '2.0.0' with 'cuda' version '11.7'. To install on Linux or Windows, you can use the following commands:

```bash
# CUDA 11.7
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia
# CUDA 11.8
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
# CPU Only
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 cpuonly -c pytorch
```

## Running Experiments

Our experiment script is `q_learning.py`. It has some required arguments while other arguments are set to default. These arguments are explained in the next section. To run one experiment, for example, you can use the following command:

```shell
python q_learning.py --EC 10 --algo DRQN --edge_id 0 --mode train --run_id 0
```

It will run DeePref at edge network to train an agent with edge capacity of '10' items. The training results will be saved under 'results' directory, and the model will be stored in 'DRQN_models' directory.

Further, we used 'codecarbon' to track carbon emissions produced by our code. To disable, please comment the related lines in 'q_learning.py' script.

### Experiment Argument Details

| Argument | Required | Default | Description |
| ------------- | ----------- | -------- | ----------- |
| `--EC` | Yes | N/A | Edge storage capacity that is containing the prefetched items. |
| `--NO_edges` | No | 3 | How many edge nodes participating in the prefetching experiment. In our experiments, we used 2 edge nodes for testing and 1 edge node for transfer learning. |
| `--algo` | Yes | DRQN | Prefetching head being used (e.g., DQN, DRQN, Belady_prefetch, top_k_size, top_k_popularity, popularity_recent, popularity_all). |
| `--RNN` | No | LSTM | Recurrent memory used for DeePref DRQN (i.e., LSTM or GRU). |
| `--approach` | No | edge | Specify which prefetching approach being used (i.e., edge or cloud). This argument is used to compare with other baselines. |
| `--body` | No | None | An embedding layer used for DeePref DRQN (i.e., LinearBody). |
| `--eviction` | No | LRU | Type of the underlying eviction algorithm being used (e.g., LRU, FIFO). |
| `--exploration` | No | epsilon | Type of exploration strategy being used by the RL agent, either DeePRef DQN or DeePref DRQN  (i.e., epislon, softmax). |
| `--run_id` | Yes | N/A | run id for each experiment. |
| `--edge_id` | Yes | N/A | edge id to use. In our experiments, we used edge#0 and edge#1 for testing, and we used edge#2 for transfer learning. |
| `--sample_size` | No | 1000 | How many time-steps per each episode or experiment. |
| `--dataset` | No | ml-100k | MovieLens Dataset ml-100k.  |
| `--way` | No | sample | Run an experiment with the entire dataset, a sample of it, or a dummy dataset. |
| `--rnn_memory` | No | sequential | Type of memory used for DeePref DQN and DeePref DRQN (i.e., sequential or random). |
| `--mode` | Yes | test | Specify which mode to use DeePref (i.e., train or test).  |



## Citing DeePref
```shell
To cite this work, please use the following bibtex:

@article{alkassab2023deepref,
  title={DeePref: Deep Reinforcement Learning For Video Prefetching In Content Delivery Networks},
  author={Alkassab, Nawras and Huang, Chin-Tser and Botran, Tania Lorido},
  journal={arXiv preprint arXiv:2310.07881},
  year={2023}
}
```

