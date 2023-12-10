import os
import json
import shutil

def get_max_step(filepath, extra_steps= 100):
    output = {}
    for category in os.listdir(filepath):
        for filename in os.listdir(os.path.join(filepath, category)):
            
            if filename.startswith("checkpoint"):
                output[category] = int(filename.split("-")[-1]) + extra_steps
    return output

language_type ={ 
    "pharo": ["classreferences", "collaborators", "example", "intent", "keyimplementationpoints", "keymessages", "responsibilities"],
    "java": ["deprecation", "expand", "ownership", "pointer", "rational", "summary", "usage"],
    "python": ["developmentnotes", "expand", "parameters", "summary", "usage"]
}


post_pretrained = False
LANGUAGE_SRC = "/cm/archive/namlh35/code-comment-classification/processed_data/valid/"
output_dir = "/cm/archive/namlh35/code-comment-classification/results/selection_with_valid/albert-datav2-class-comment"
max_step_src = "/cm/archive/namlh35/code-comment-classification/results/selection_with_valid/roberta-postpretrained-datav2-class-comment"
max_step_src = None
if max_step_src is not None:
    max_step_dict = get_max_step(max_step_src)

if not os.path.exists(output_dir):
    os.mkdir(output_dir)


run_command = """CUDA_VISIBLE_DEVICES=0,1 python3 training/run.py \
    --seed 0 \
    --model_short_name albert \
    --mix_type HSUM \
    --count 4 \
    --model_name_or_path albert-base-v2 \
    --train_file {} \
    --validation_file {} \
    --test_file {} \
    --output_dir {} \
    --text_column_names class,comment_sentence \
    --label_column_name instance_type \
    --metric_for_best_model f1 \
    --metric_name f1 \
    --text_column_delimiter "</s>" \
    --max_seq_length 64 \
    --per_device_train_batch_size 32 \
    --learning_rate 1e-5 \
    --num_train_epochs {} \
    --max_steps {} \
    --do_train \
    --do_predict \
    --load_best_model_at_end \
    --evaluation_strategy steps \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit {} \
    --overwrite_output_dir
"""

if not post_pretrained:
    for lang in os.listdir(LANGUAGE_SRC):
        if lang == "java":
            num_epoch = 10
        else:
            num_epoch = 20
        for comt_type in language_type[lang]:
            category = lang + "_" + comt_type
            if max_step_src is not None:
                optimal_step = max_step_dict[category]
            else:
                optimal_step = -1
            
            if "novalid" in LANGUAGE_SRC:
                valid_name = "train.csv"
                save_total_limit = -1
            else:
                valid_name = "valid.csv"
                save_total_limit = 1

            if not os.path.exists(os.path.join(output_dir, category)):
                print("Training")
                os.system(run_command.format(os.path.join(LANGUAGE_SRC, lang, comt_type, "train.csv"),
                    os.path.join(LANGUAGE_SRC, lang, comt_type, valid_name),
                    os.path.join(LANGUAGE_SRC, lang, comt_type, "test.csv"),
                    output_dir,
                    num_epoch,
                    -1,
                    save_total_limit
                ))
            #Remove unwanted checkpoints
            if max_step_src is not None:
                all_ckpt = [int(x.split("-")[-1]) for x in os.listdir(os.path.join(output_dir, category)) if x.startswith("checkpoint")]
                optimal_ckpt = max(all_ckpt) if max(all_ckpt) < optimal_step else optimal_step # sometime due to extra_steps => optimal_step > max_step
                max_step_dict[category] = optimal_ckpt
                for dirname in os.listdir(os.path.join(output_dir, category)):
                    if dirname != f"checkpoint-{optimal_ckpt}":
                        if os.path.isdir(os.path.join(output_dir, category, dirname)):
                            shutil.rmtree(os.path.join(output_dir, category, dirname), ignore_errors=True)
                        else:
                            os.remove(os.path.join(output_dir, category, dirname))
else:
    os.system(run_command.format(os.path.join(LANGUAGE_SRC, "train.csv"),
            os.path.join(LANGUAGE_SRC, "test.csv"),
            os.path.join(LANGUAGE_SRC, "test.csv"),
            output_dir,
            10,
            -1,
            -1
            ))

if max_step_src is not None:
    with open(os.path.join(output_dir, "optimal_step.json"), "w") as f:
        json.dump(max_step_dict, f, indent=4)
