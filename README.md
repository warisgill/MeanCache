>**Note:** This repository contains code for MeanCache, released in response to multiple community requests. While the codebase is not yet fully portable or executable in all environments—due to hardcoded paths and some experimental components—it remains useful for exploration and research purposes. 


## Dataset Information

- The contextual queries datasets are located in the `dataset_contextual_queries` directory
- Additional datasets used in experiments are sourced from publicly available repositories. Please use them from their original locations to ensure compliance with their licenses and terms of use.



## Repository Status

Due to time constraints (i am busy with my internships), the code has some limitations:

- The implementation contains hardcoded paths that may need adjustment for different environments
- The `diskcache` library is used extensively for storing models and intermediate results, with specific cache keys that are referenced throughout the code that only work in my environment. 
- Some parts of the codebase may not be directly executable without modifications
- The repository includes some experimental/unnecessary code.




## Future Work on This Repository

I am providing this code in its current state to allow others to explore the MeanCache implementation. After completing my internship, I plan to:

1. Refactor the codebase to remove hardcoded paths
2. Provide better documentation and usage examples
3. Create a fully executable artifact with clear setup instructions
4. Simplify the dependencies and configuration process

> **Summary:** After my Microsoft internship and when I have some free time, I will work on improving the codebase to make it executable with just a single click, similar to my other artifacts like [TraceFL](https://github.com/SEED-VT/TraceFL) and [FedDebug](https://flower.ai/docs/baselines/feddebug.html). **Apologies for any inconvenience and thank you for your understanding.**


## Other Work on This Topic

During my **Redis** internship, I trained embedding models for semantic caching ([redis/langcache-embed-v1](https://huggingface.co/redis/langcache-embed-v1) and [redis/langcache-embed-v2](https://huggingface.co/redis/langcache-embed-v2)), reaching thousands of downloads on Huggingface. The corresponding research paper is [Advancing Semantic Caching for LLMs with Domain-Specific Embeddings and Synthetic Data](https://arxiv.org/pdf/2504.02268). This might be useful for those interested in semantic caching techniques and embedding models.

