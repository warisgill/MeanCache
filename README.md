>**Note:** This repository contains code for MeanCache. The codebase has hardcoded paths (local to my enviorment) and some other experimental components. I will update these paths in the future to make it more user-friendly and easier to run without modifications. 


## Dataset Information

- The contextual queries datasets are located in the `dataset_contextual_queries` directory
- Additional datasets used in experiments are sourced from publicly available repositories. Please use them from their original locations to ensure compliance with their licenses and terms of use.


# Code Informtion

- `generate_scripts.py`: This script generates the necessary scripts for training and evaluating the MeanCache model.
- `fl_sim_train.py`: This script is used for training the MeanCache model using Federated Learning simulation.
- `cache_comparison.py` and `eval.py`: Contains basic cache comparison functions. 
- `utils.py`: Contains utility functions used across the codebase.
- `logs/`: This directory contains logs generated during the training and evaluation processes.
- `run.sh`: contains configurations for running the training the scripts. 



## Other Work on This Topic

During my **Redis** internship, I trained embedding models for semantic caching ([redis/langcache-embed-v1](https://huggingface.co/redis/langcache-embed-v1) and [redis/langcache-embed-v2](https://huggingface.co/redis/langcache-embed-v2)), reaching thousands of downloads on Huggingface. The corresponding research paper is [Advancing Semantic Caching for LLMs with Domain-Specific Embeddings and Synthetic Data](https://arxiv.org/pdf/2504.02268). This might be useful for those interested in semantic caching techniques and embedding models.

