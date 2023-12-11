from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from custom_model import CodeBERTHSUMForSequenceClassification, CodeBERTForSequenceClassification
import json, os
import pandas as pd
import time
from datasets import Dataset
from typing import *
from datasets import load_dataset, get_dataset_split_names
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', default="./code-comment-classification/stage1/Dopamin")
parser.add_argument('--model_name', default="codebert-hsum")
parser.add_argument('--model_path', default="./models/Dopamin")
parser.add_argument('--test_src', default=None)

args = parser.parse_args()


def get_model_class(model_name):
    if "hsum" in model_name:
        return CodeBERTHSUMForSequenceClassification
    elif "-custom" in model_name:
        return CodeBERTForSequenceClassification
    else:
        return AutoModelForSequenceClassification

def get_precision_recall_f1(tp: float, fp: float, fn: float) -> Tuple[float, float, float]:
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall / (precision + recall))
    return precision, recall, f1

model_name = args.model_name
model_src = args.model_path
test_src= args.test_src
# test_src = None
if not test_src:
    with open(os.path.join(model_src, "optimal_step.json")) as f:
        optimal_step_metric = json.load(f)

langs = ['java', 'python', 'pharo']
lan_cats = []
datasets = {}
for lan in langs: # for each language
    if test_src is not None: # for evaluation on validation set
        concat_df = []
        for cate in os.listdir(os.path.join(test_src, lan)):
            concat_df.append(pd.read_csv(os.path.join(test_src, lan, cate, "valid.csv")))
        df = pd.concat(concat_df)
    else: # for evaluation on test set
        df = pd.read_csv(f'./{lan}/input/{lan}.csv')
    df['combo'] = df[['class', 'comment_sentence']].agg('</s>'.join, axis=1)
    df['label'] = df.instance_type
    df["combo_len"] = [len(x.split()) for x in df["combo"]]
    cats = list(map(lambda x: lan + '_' + x, list(set(df.category))))

    for cat in list(set(df.category)): # for each category
        filtered =  df[df.category == cat].sort_values(by='combo_len')
        train_data = Dataset.from_pandas(filtered[filtered.partition == 0])
        if test_src:
            test_data = Dataset.from_pandas(filtered)
        else:
            test_data = Dataset.from_pandas(filtered[filtered.partition == 1])
        datasets[f'{lan}_{cat}'.lower()] = {'train_data': train_data, 'test_data' : test_data}
        lan_cats.append(f'{lan}_{cat}'.lower())

def get_prediction(x, model, tokenizer, num_iter):
    y_hat = []
    for i in range(int(num_iter)):
        inputs = tokenizer(x[i*batch_size:(i+1)*batch_size], max_length=32, padding=True, truncation=True, return_tensors="pt")
        inputs = {k:v.cuda() for k,v in inputs.items()}
        logits = model(**inputs).logits
        y_hat.extend(np.argmax(logits.cpu().numpy(), axis=1).tolist())
    return y_hat

scores = []
batch_size = 64

for lan_cat in lan_cats:
    # if lan_cat not in ["java_deprecation", "pharo_classreferences", "python_parameters"]:
    #   continue
    # load models and data
    print(lan_cat)
    if not test_src:
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_src,lan_cat,"checkpoint-{}".format(optimal_step_metric[lan_cat])))
        model = get_model_class(model_name).from_pretrained(os.path.join(model_src,lan_cat,"checkpoint-{}".format(optimal_step_metric[lan_cat])))
    else:
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_src,lan_cat))
        model = get_model_class(model_name).from_pretrained(os.path.join(model_src,lan_cat)) 
    # model.cuda()
    # model.eval()
    test_data = datasets[lan_cat]['test_data']
    x = test_data["combo"]
    y = test_data['label']
    num_iter = np.ceil(len(x) / batch_size)

    # # run and time 10 times for each cat
    with torch.no_grad():
      for it in range(1):
        ############# TIME BLOCK #####################
        num_iter = np.ceil(len(x) / batch_size)
        start = time.time()
        y_hat = get_prediction(x, model, tokenizer, num_iter)
        elapsed_time = time.time() - start

        time_per_sample = elapsed_time / len(y)

        # # ############# TIME BLOCK #####################
        _, fp, fn, tp = confusion_matrix(y_hat, y).ravel()
        wf1 = f1_score(y, y_hat, average='weighted')
        precision, recall, f1 = get_precision_recall_f1(tp, fp, fn)
        scores.append({'lan_cat': lan_cat.lower(),'precision': precision,'recall': recall,'f1': f1,'wf1': wf1, 'avg_runtime': time_per_sample, 'iteration': it, 'len': len(test_data)})


df = pd.DataFrame(scores).groupby('lan_cat').mean().reset_index()
df['time_std'] = pd.DataFrame(scores).groupby('lan_cat').std()['avg_runtime']
print(f"Average runtime: {round(df['avg_runtime'].mean(), 5)}")
print(f"Average f1: {round(df['f1'].mean(), 2)}")
df.precision = df.precision.round(2)
df.recall = df.recall.round(4)
df.f1 = df.f1.round(4)
df.avg_runtime = df.avg_runtime.round(4)

df = df[['lan_cat', 'precision','recall', 'f1', 'avg_runtime']]
df2 = {'lan_cat': "Average",'precision': round(df['precision'].mean(),4),'recall': round(df['recall'].mean(),4),'f1': round(df['f1'].mean(), 4), 'avg_runtime': round(df['avg_runtime'].mean(), 5)}
df = pd.concat([df, pd.DataFrame([df2])], ignore_index=True)
print(df)
df.to_csv(os.path.join(model_src, "results.csv"), index=False)
