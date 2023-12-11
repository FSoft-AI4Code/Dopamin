# Dopamin: Transformer-based Comment Classifiers through Domain Post-training and Multi-level layer aggregation
This repository includes our implementation for training, testing, and utilizing Dopamin, which is our submission for [NLBSE'24 Tool Competition: Code Comment Classification](https://nlbse2024.github.io/tools/).

# Quickstart Guide
## Set up
Install requirements: ```pip install -r requirements.txt```

Download dataset: ```git clone https://github.com/nlbse2024/code-comment-classification.git```

## Data preparation
Create data for the post-training stage: 
```python
python process_data.py --save_dir ./code-comment-classification/processed_data/all --post_training
```

Create training and evaluation set: 
```python
python process_data.py --save_dir ./code-comment-classification/processed_data/valid --validation
```

Original_data: 
```
python process_data.py --save_dir ./code-comment-classification/processed_data/novalid
```

## Training
All training and evaluation scripts can be found in [training Dopamin](https://github.com/FSoft-AI4Code/Dopamin/tree/main/training)

### Post-training stage
```python
python training/autorun.py --output_dir ./models/Dopamin_post_training --post_training
```
You can download the post-trained model at https://huggingface.co/Fsoft-AIC/dopamin-post-training

### Training Dopamin for each category
1. Training model with validation set to obtain the best checkpoint step
```python
python training/autorun.py --output_dir ./models/Dopamin_valid --validation
```
2. Training model with original training data with the found optimal step
```python
python training/autorun.py --output_dir ./models/Dopamin --optimal_step_file ./models/Dopamin_valid
```

## Evaluation
To run the evaluation of Dopamin, please refer to the [evaluation notebook](https://github.com/FSoft-AI4Code/Dopamin/blob/main/Dopamin_evaluation.ipynb)
or if you want to use the script:
```python
python training/predict.py --model_name codebert-hsum \
                           --model_path ./models/Dopamin \
```                      

All model checkpoints are publicity available at [Huggingface Hub - Dopamin](https://huggingface.co/collections/Fsoft-AIC/dopamin-6575bdeb7068a850897e4404) for replication purposes.

# Citation
```bibtex
@software{
  Dopamin_2024,
  author = {Hai, Nam Le and Bui, Nghi DQ},
  year = {2024},
  title = {Dopamin: Transformer-based Comment Classifiers through Domain Post-training and Multi-level layer aggregation},
  url = {https://github.com/FSoft-AI4Code/Dopamin},
  huggingface= {https://huggingface.co/collections/Fsoft-AIC/dopamin-6575bdeb7068a850897e4404}
}
```
