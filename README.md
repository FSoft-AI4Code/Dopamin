# Dopamin: Transformer-based Comment Classifiers through Domain Post-training and Multi-level layer aggregation
This repository includes our implementation for training, testing, and utilizing Dopamin, which is our submission for [NLBSE'24 Tool Competition: Code Comment Classification](https://nlbse2024.github.io/tools/).

# Quickstart Guide
## Preparation
Install requirements: ```pip install -r requirements.txt```
Download dataset: ```git clone https://github.com/nlbse2024/code-comment-classification.git```

## Data process
Create data for the post-training stage: ```python process_data.py --save_dir ./code-comment-classification/processed_data/all --post_training```
Create training and evaluation set: ```python process_data.py --save_dir ./code-comment-classification/processed_data/valid --validation```
Original_data: ```python process_data.py --save_dir ./code-comment-classification/processed_data/novalid```

# Evaluation
To run the evaluation of Dopamin, please refer to the [evaluation notebook](https://github.com/FSoft-AI4Code/Dopamin/blob/main/Dopamin_evaluation.ipynb)

All model checkpoints are publicity available at Huggingface Hub - [Dopamin](https://huggingface.co/collections/Fsoft-AIC/dopamin-6575bdeb7068a850897e4404) for replication purposes.
