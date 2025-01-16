import os
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd

import datasets
import dataeval.w_mbpp as mbpp
import dataeval.w_repoeval as repo_eval
import argparse

import dataeval.w_humaneval as human_eval
import dataeval.w_mbpp as mbpp
import dataeval.w_ds1000 as ds1000
import dataeval.w_repoeval as repo_eval
import dataeval.w_deveval as dev_eval

def get_dataset_fn(data_name):
    if data_name == 'human_eval':
        return human_eval.get_dataset
    if data_name == 'mbpp':
        return mbpp.get_dataset
    if data_name == 'ds1000':
        return ds1000.get_dataset
    if data_name == 'repo_eval':
        return repo_eval.get_dataset
    if data_name == 'dev_eval':
        return dev_eval.get_dataset

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    default="deepseek-ai/deepseek-coder-1.3b-instruct",
    help="which results to run",
)
parser.add_argument(
    "--resume",
    action="store_true",
    default=False,
    help="where to resume inference",
)
args = parser.parse_args()
model_name = args.model

model = AutoModelForCausalLM.from_pretrained(
    model_name, resume_download=True, trust_remote_code=True
)
model.to("cuda")

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# Need to set the padding token to the eos token for generation
if tokenizer.eos_token:
    tokenizer.pad_token = tokenizer.eos_token
else:
    tokenizer.add_special_tokens({
        "pad_token": "<pad>"
    })

import random
random.seed(37)

# Generate unique numbers within the range
train_task_ids = random.sample(range(11, 510), 450)

# df = pd.read_parquet("/drive2/tuandung/WCODELLM/LFCLF_embedding_ds1000_deepseek-ai_deepseek-coder-1.3b-instruct_24_label_cleaned_code.parquet")

with open('/drive2/tuandung/WCODELLM/benchmark/DevEval/data/train_project_ids.txt', 'r') as f:
    train_project_ids = f.readlines()
# df_test = df[~df['task_id'].isin(train_task_ids)]
# # df_test = df
# df_test_dict = df_test.to_dict(orient='records')

def build_prompt(Query):
    return f"Are you capable of providing an accurate response to the query given below? Respond only to this question with ’yes’ or ’no’ and do not address the content of the query itself. The query in block [Query] and [/Query] and your respone after 'Answer'. \n[Query]\n{Query}\n[/Query] \n\nAre you capable of providing an accurate response to the query given above without more information? Respond only to this question with yes or no. \nAnswer: "
# \n\nQuery: Write a function to find the similar elements from the given two tuple lists. \nAnswer: yes
# repoeval_ds = repo_eval.get_dataset(tokenizer, "python", sft=False, instruction=True)
problem_file = '/drive2/tuandung/WCODELLM/benchmark/MBPP/data/mbpp.jsonl'

def build_prompt_with_output(Query, Respone):
    return f"The query in block [Query] and [/Query] and your respone in block [Respone] and [/Respone]. \n[Query]\n{Query}\n[/Query] \n[Respone]\n{Respone}\n[/Respone]\n\nIs your respone is accurate to query? Answer only to this question with yes or no. \nAnswer: "

# problem_file = '/drive2/tuandung/WCODELLM/benchmark/HumanEval/data/humaneval-python.jsonl'
# problem_file = '/drive2/tuandung/WCODELLM/benchmark/DS_1000/data/ds1000.jsonl'
# problem_file = '/drive2/tuandung/WCODELLM/benchmark/DevEval/data/completion_dataset.jsonl'
# # sequences = pd.read_pickle(continue_from)
# examples = [json.loads(x) for x in open(problem_file) if x.strip()]
# mbpp_ds = get_dataset_fn('mbpp')(tokenizer, language='python', instruction=True)
dev_eval_ds = get_dataset_fn('dev_eval')(tokenizer, language='python', instruction=True)

namespace_to_project = {}
# project_counts = {}
data_file = '/drive2/tuandung/WCODELLM/benchmark/DevEval/data/data_clean.jsonl'
with open(data_file, 'r') as f:
    for line in f:
        js = json.loads(line)
        namespace = js['namespace']
        ns = js['project_path']
        # project_counts[ns] = project_counts.get(ns, 0) + 1
        namespace_to_project[namespace] = ns
        
# Apply padding on the left since we are doing generation
padding_side_default = tokenizer.padding_side
tokenizer.padding_side = "left"

baseline_prompts = []
dev_eval_ds = list(dev_eval_ds)[:5]
for test_dict in tqdm(dev_eval_ds):
    if namespace_to_project[test_dict['task_id']] in train_project_ids:
        continue
    prompt = test_dict['prompt']
    # respone = test_dict['cleaned_code']
#     prompt = '''
# Please continue to complete the function. You are not allowed to modify the given code and do the completion only. Please return all completed function in a codeblock. Here is the given code to do completion:
# ```{}
# {}
# ```
# '''.strip().format('python', test_dict['prompt'].strip())
    baseline_prompts.append((test_dict['task_id'], build_prompt(prompt)))

# baseline_prompts = [build_prompt(prompt) for prompt in df_test_dict['prompt']]

# df_test = pd.DataFrame(df_test_dict)

# Tokenize each batch
# Put back the original padding behavior
tokenizer.padding_side = padding_side_default

model_name = model_name.replace('/', '-')
generation_config = {
    "do_sample": False,
    "max_new_tokens": 32,
    "num_beams": 1
}

generated_texts = []

# with tqdm(total=len(df_test)) as pbar:
for prompt_tuple in tqdm(baseline_prompts):
    # prompt = build_prompt(p)
    # print(prompt)
    task_id, prompt = prompt_tuple
    # tokenized_prompt = tokenizer(prompt, padding=True,
    #                                 return_tensors="pt").to("cuda")
    inputs = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': prompt }],
        return_tensors="pt", add_generation_prompt=True
    ).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            # **tokenized_prompt,
            **generation_config
        )
    generated_text = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
    # pbar.update(1)
    print(generated_text)
    label = 0
    if 'yes' in generated_text or 'Yes' in generated_text:
        label = 1
        
    generated_texts.append((task_id, label, generated_text))

with open(f'output/{model_name}_dev_eval_baseline.jsonl', 'w') as f:
    for task_id, label, generated_text in generated_texts:
        f.write(json.dumps({
            'task_id': task_id,
            'label': label,
            'generated_text': generated_text
        }) + '\n')