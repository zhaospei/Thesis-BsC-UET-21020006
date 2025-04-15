import os
import pickle
import argparse
import tqdm
import torch
import pandas as pd

import models
import dataeval.w_humaneval as human_eval
import dataeval.w_mbpp as mbpp
import dataeval.w_ds1000 as ds1000
import dataeval.w_repoeval as repo_eval
import dataeval.w_evocodebench as evocodebench
import dataeval.w_repoexec as repo_exec
import dataeval.w_deveval as dev_eval
from dataeval.w_humaneval import extract_generation_code as human_eval_egc
from dataeval.w_mbpp import extract_generation_code as mbpp_eval_egc
from dataeval.w_ds1000 import extract_generation_code as ds1000_eval_egc
from dataeval.w_evocodebench import extract_generation_code as evocodebench_eval_egc
from dataeval.w_repoeval import extract_generation_code as repoeval_eval_egc
from dataeval.w_deveval import extract_generation_code as deveval_eval_egc

from func.metric import *


parser = argparse.ArgumentParser()
parser.add_argument('--generate_dir', type=str, default='generate')
# parser.add_argument('--output_dir', type=str, default='output')
parser.add_argument('--dataset', type=str, default='human_eval')
parser.add_argument('--model_name', type=str, default='opt-13b')
parser.add_argument("--language", default="python", type=str,)
parser.add_argument("--layers", default=[-1], type=int, nargs='+')
parser.add_argument("--type", default="LFCLF", type=str)
args = parser.parse_args()
   

def get_dataset_fn(data_name):
    if data_name == 'human_eval':
        return human_eval.get_dataset
    if data_name == 'mbpp':
        return mbpp.get_dataset
    if data_name == 'ds1000':
        return ds1000.get_dataset
    if data_name == 'repo_eval':
        return repo_eval.get_dataset
    if data_name == 'evocodebench':
        return evocodebench.get_dataset
    if data_name == 'repoexec':
        return repo_exec.get_dataset
    if data_name == 'dev_eval':
        return dev_eval.get_dataset
    raise ValueError(f"Unknown dataset {data_name}")

def extract_generation_code_fun(data_name):
    if data_name == 'human_eval':
        return human_eval_egc
    if data_name == 'mbpp':
        return mbpp_eval_egc
    if data_name == 'ds1000':
        return ds1000_eval_egc
    if data_name == 'repo_eval':
        return repoeval_eval_egc
    if data_name == 'evocodebench':
        return evocodebench_eval_egc
    if data_name == 'repoexec':
        return repoeval_eval_egc
    if data_name == 'dev_eval':
        return deveval_eval_egc

def process_lfclf():
    tokenizer = models.load_tokenizer(args.model_name)
    if 'chat' or 'instruct' in args.model_name.lower():
        instruction = True
    else:
        instruction = False
    dataset = get_dataset_fn(args.dataset)(tokenizer, language=args.language, instruction=instruction)
    dataset_egc = extract_generation_code_fun(args.dataset)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    
    output_dir = args.generate_dir.replace('temp', 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    clean_generations_range_all = {}
    
    for example in tqdm.tqdm(dataset, total=len(dataset)):
        has_error = False
        task_id_path =  str(example['task_id']).replace('/','_').replace('[','_').replace(']','_')
        if args.dataset == 'mbpp' or args.dataset == 'ds1000':
            task_id_path = f'tensor({task_id_path})'
        task_generation_seqs_path = f'generation_sequences_output_{task_id_path}.pkl'
        task_generation_seqs_path = os.path.join(args.generate_dir, task_generation_seqs_path)
        # print(task_generation_seqs_path)
        if not os.path.exists(task_generation_seqs_path):
            print(f'File {task_id_path} not found. Skipping...')
            continue
        
        # print(f'Found {task_id_path}. Processing...')
        
        with open(task_generation_seqs_path, 'rb') as f:
            task_generation_seqs = pickle.load(f)
        
        clean_generations_range = []
        for generated_ids in task_generation_seqs['generations_ids']:
            gen = tokenizer.decode(generated_ids, skip_special_tokens=True)
            clean_generation_decoded = dataset_egc(example, gen, args.language)
            start_ind, end_ind = getCleanGenerationRange(generated_ids.tolist(), clean_generation_decoded, tokenizer)
            if start_ind is None or end_ind is None:
                has_error = True
                # print("gen:", gen)
                # print("clean_generation_decoded:", clean_generation_decoded)
                print(f'Cannot find clean generation range for {task_id_path}')
                start_ind, end_ind = getGenerationRange(generated_ids.tolist(), tokenizer)
                clean_generations_range.append((start_ind, end_ind, has_error))
            else:
                clean_generations_range.append((start_ind, end_ind, has_error))

        clean_generations_range_all[task_id_path] = clean_generations_range
    # print(clean_generations_range_all)
    for layer in args.layers:
        print(f'Processing layer {layer}')
        results = pd.DataFrame(columns=[
            "task_id", 
            "completion_id", 
            "num_tokens", 
            "generation", 
            "first_token_embedding", 
            "last_token_embedding",
            "first_token_code_embedding",
            "last_token_code_embedding",
            "has_error",
            "first_code_index",
            "last_code_index",
            "first_index",
            "last_index",
            "generated_ids"
        ])
        found_sample = 0
        for example in tqdm.tqdm(dataset, total=len(dataset)):
            task_id_path =  str(example['task_id']).replace('/','_').replace('[','_').replace(']','_')
            if args.dataset == 'mbpp' or args.dataset == 'ds1000':
                task_id_path = f'tensor({task_id_path})'
            task_generation_seqs_path = f'generation_sequences_output_{task_id_path}.pkl'
            task_generation_seqs_path = os.path.join(args.generate_dir, task_generation_seqs_path)
            # if task_id_path == 'arctic.hooks.register_get_auth_hook':
            #     break
            if not os.path.exists(task_generation_seqs_path):
                # print(f'File {task_id_path} not found. Skipping...')
                continue
            with open(task_generation_seqs_path, 'rb') as f:
                task_generation_seqs = pickle.load(f)
            found_sample += 1
            clean_generations_range = clean_generations_range_all[task_id_path]
            task_embedding_path = f'all_token_embedding_{task_id_path}_{layer}.pkl'
            task_embedding_path = os.path.join(args.generate_dir, task_embedding_path)
            if not os.path.exists(task_embedding_path):
                print(f'File {task_id_path} {layer} not found. Skipping...')
                continue
            
            with open(task_embedding_path, 'rb') as f:
                task_embedding = pickle.load(f)
            
            task_last_token_embedding = []
            for j in range(len(task_generation_seqs['generations'])):
                task_id = example['task_id']
                completion_id = str(task_id) + '_' + str(j)
                # num_tokens = task_generation_seqs['num_tokens'][j]
                generation = task_generation_seqs["generations"][j]
                generated_ids = task_generation_seqs["generations_ids"][j]
                start_code_ind, end_code_ind, has_error = clean_generations_range[j]
                start_ind, end_ind = getGenerationRange(generated_ids.tolist(), tokenizer)
                num_tokens = end_ind - start_ind
                layer_embedding = task_embedding['layer_embeddings'][j]

                extracted_code = tokenizer.decode(generated_ids.tolist()[start_code_ind:end_code_ind], skip_special_tokens=True)

                start_ind = max(0, start_ind - 1)
                end_ind = end_ind - 1
                start_code_ind = max(0, start_code_ind)
                end_code_ind = end_code_ind - 1
                
                try:
                    first_token_embedding = layer_embedding[start_ind].tolist()
                    last_token_embedding = layer_embedding[end_ind - 1].tolist()
                    first_token_code_embedding = layer_embedding[start_code_ind].tolist()
                    last_token_code_embedding = layer_embedding[end_code_ind - 1].tolist()

                    results = results._append({
                        "task_id": task_id, 
                        "completion_id": completion_id,
                        "num_tokens": num_tokens,
                        "generation": generation, 
                        "first_token_embedding": first_token_embedding, 
                        "last_token_embedding": last_token_embedding,
                        "first_token_code_embedding": first_token_code_embedding,
                        "last_token_code_embedding": last_token_code_embedding,
                        "has_error": has_error,
                        "extracted_code": extracted_code,
                        "first_code_index": start_code_ind,
                        "last_code_index" : end_code_ind - 1,
                        "first_index": start_ind,
                        "last_index": end_ind - 1,
                        "generated_ids": generated_ids.tolist()
                    }, 
                    ignore_index=True)
                except:
                    print('Cant extract layer embedding')

        print(f'Found {found_sample} / {len(dataset)}')
        model_name = args.model_name.replace('/', '_')
        results.to_parquet(os.path.join(output_dir, f'LFCLF_embedding_{args.dataset}_{model_name}_{layer}.parquet'))
            
    return


if __name__ == '__main__':
    if args.type == 'LFCLF':
        process_lfclf()
    else:
        raise ValueError(f"Unknown type {args.type}")