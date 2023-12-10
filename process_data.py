import pandas as pd
import os
from sklearn.model_selection import train_test_split
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--src_data', default="./code-comment-classification")
parser.add_argument('--save_dir', default="./code-comment-classification/processed_data/")
parser.add_argument('--post_training', action="store_true")
parser.add_argument('--validation', action="store_true")
args = parser.parse_args()


src = args.src_data
save_dir = args.save_dir
languages = ["python", "java", "pharo"]
combined = args.post_training

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

if combined:
    train_set, test_set = [], []
for language in languages:  
    df = pd.read_csv(os.path.join(src,language,f"input/{language}.csv"))
    df["category"] = df["category"].str.lower()
    df['comment_sentence'] = df['comment_sentence'].apply(lambda x: x.strip())
    if combined:
        train_set.append(df[df.partition == 0])
        test_set.append(df[df.partition == 1])      
    else:
        for cat in list(set(df.category)): # for each category
            if not os.path.exists(os.path.join(save_dir, language, cat)):
                os.makedirs(os.path.join(save_dir, language, cat))
            filtered =  df[df.category == cat]
            train_data = filtered[filtered.partition == 0]
            test_data = filtered[filtered.partition == 1]

            if args.validation:
                train_data, valid_data = train_test_split(train_data, stratify=train_data["instance_type"], test_size=0.1, random_state=0)
                valid_data.to_csv(os.path.join(save_dir, language, cat, "valid.csv"), index=False)

            train_data.to_csv(os.path.join(save_dir, language, cat, "train.csv"), index=False)
            test_data.to_csv(os.path.join(save_dir, language, cat, "test.csv"), index=False)
            

if combined:
    train_data = pd.concat(train_set, ignore_index= True)
    test_data = pd.concat(test_set, ignore_index= True)
    train_data.to_csv(os.path.join(save_dir, "train.csv"), index=False)
    test_data.to_csv(os.path.join(save_dir, "test.csv"), index=False)