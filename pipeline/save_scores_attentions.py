import argparse
import glob
import json
import os
import copy
import time
import gc
import pandas as pd
import numpy as np
import torch
import tqdm
import pickle
from transformers import StoppingCriteria, StoppingCriteriaList
from sentence_transformers import SentenceTransformer
from torchmetrics.text.bert import BERTScore

import _settings
import dataeval.w_humaneval as human_eval
import dataeval.w_mbpp as mbpp
import dataeval.w_ds1000 as ds1000
import dataeval.w_repoeval as repo_eval
import dataeval.w_evocodebench as evocodebench
import dataeval.w_repoexec as repo_exec
import dataeval.w_deveval as dev_eval
from dataeval.w_humaneval import cleanup_code as human_eval_cleanup_code
from dataeval.w_humaneval import extract_generation_code as human_eval_egc
from dataeval.w_mbpp import extract_generation_code as mbpp_eval_egc
from dataeval.w_ds1000 import extract_generation_code as ds1000_eval_egc
from dataeval.w_evocodebench import extract_generation_code as evocodebench_eval_egc
from dataeval.w_repoeval import extract_generation_code as repoeval_eval_egc
from dataeval.w_deveval import extract_generation_code as deveval_eval_egc
import models
import utils
from func.metric import *

passed_input_len_task = ['repo_eval', 'evocodebench', 'repoexec', 'dev_eval']

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='llama-13b-hf')
parser.add_argument('--dataset', type=str, default='human_eval')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--tensor_parallel_size', type=int, default=1)
parser.add_argument('--fraction_of_data_to_use', type=float, default=1.0)
parser.add_argument('--num_generations_per_prompt', type=int, default=10)
parser.add_argument('--max_num_gen_once', type=int, default=5)
parser.add_argument('--max_new_tokens', type=int, default=500)
parser.add_argument('--temperature', type=float, default=1.0)
parser.add_argument('--decoding_method', type=str, default='greedy')
parser.add_argument('--top_p', type=float, default=1.0)
parser.add_argument('--top_k', type=int, default=10)
parser.add_argument('--seed', type=int, default=2023)
parser.add_argument('--nprocess', type=int, default=None)
parser.add_argument('--project_ind', type=int, default=0)
parser.add_argument("--layers", default=-1, nargs='*', type=int,
                        help="List of layers of the LM to save embeddings from indexed negatively from the end")
parser.add_argument("--language", default="python", type=str,)
parser.add_argument("--load_in_8bit", action="store_true", help="Whether to load the model in 8bit mode")

args = parser.parse_args()
print(args)
print(args.model.replace('/', '_'))
ml_time = int(time.time() * 1000)
layer_name = '_'.join(str(x) for x in args.layers)
OUTPUT_DIR = os.path.join(_settings.GENERATION_FOLDER, f'att_hidden_states_{args.model.replace("/", "_")}_{args.dataset}_{args.language}_{layer_name}')
os.makedirs(OUTPUT_DIR, exist_ok=True)
logInfo = open(os.path.join(OUTPUT_DIR, "logInfo.txt"), mode="w",encoding="utf-8")

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

class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords_str, tokenizer):
        StoppingCriteria.__init__(self)
        self.current_context = []
        self.tokenizer = tokenizer
        self.keywords_str = keywords_str
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        self.current_context.append(input_ids[0][-1].item())
        current_context = self.tokenizer.decode(self.current_context)
        for word in self.keywords_str:
            if word in current_context:
                return True
        return False

# _UNUSED_TOKENIZER = models.load_tokenizer()
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


@torch.no_grad()
def get_generations(model_name:str, args, seed=1, old_sequences=None, max_num_gen_once=args.max_num_gen_once,cache_dir='output'):
    device = args.device
    model, tokenizer = models.load_model_and_tokenizer(model_name, args.device, args.load_in_8bit)    
    utils.seed_everything(seed)
    print(model)
    model.eval()
    dataset_egc = extract_generation_code_fun(args.dataset)
    if 'chat' or 'instruct' in model_name.lower():
        instruction = True
    else:
        instruction = False
    dataset = get_dataset_fn(args.dataset)(tokenizer, language=args.language, instruction=instruction)
    if hasattr(dataset[0],'stopwords'):
        stop_words = dataset[0]['stopwords']
    else:
        stop_words = []
    if args.fraction_of_data_to_use < 1.0:
        dataset = dataset.train_test_split(test_size=(1 - args.fraction_of_data_to_use), seed=seed)['train']
    # dataset = list(dataset)[-250:]
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    
    print('len dataset', len(dataloader))
    if old_sequences is None:
        old_sequences = []
    old_sequences = {_['task_id']: _ for _ in old_sequences}
    sequences = {}
    # for layer in args.layers:
    #     sequences[layer] = []
    generation_sequences_output = []
    
    for batch_idx, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
        # print(batch.keys())
        task_id_path = str(batch['task_id'][0]).replace('/','_').replace('[','_').replace(']','_')
        out_dir_task_id = os.path.join(cache_dir, f"{task_id_path}.pkl")
        if batch['task_id'][0] in old_sequences:
            sequences.append(old_sequences[batch['task_id'][0]])
            continue
        # if os.path.exists(os.path.join(cache_dir, f'generation_sequences_output_{task_id_path}.pkl')):
        #     print(f'Generated {task_id_path}!')
        #     continue # generated
        # else:
        #     print(f'Processing {task_id_path} ...')
        input_ids = batch['input_ids'].to(device)
        print(f"input_ids shape: {input_ids.shape}")
        if args.dataset not in passed_input_len_task  and (input_ids.shape[-1] >1000 or input_ids.shape[-1] < 9):
            continue
        input_length = input_ids.shape[1]
        torch.cuda.empty_cache()
        
        generations = []
        generations_decoded = []
        all_scores_softmax = []
        # print("Prompt:", tokenizer.decode(input_ids.cpu()[0], skip_special_tokens=True))
        num_gens = args.num_generations_per_prompt
        all_token_hidden_states_layer_list = {}
        off_set = 0
        while num_gens > 0:
            dict_outputs =  model.generate(input_ids, attention_mask=batch['attention_mask'].to(device),
                            num_beams=1, max_new_tokens=args.max_new_tokens, num_return_sequences=min(max_num_gen_once, num_gens),
                            do_sample=True, 
                            top_p=args.top_p, 
                            top_k=args.top_k,
                            temperature=args.temperature, 
                            eos_token_id=tokenizer.eos_token_id,
                            pad_token_id=tokenizer.eos_token_id,
                            output_hidden_states=True, 
                            return_dict_in_generate=True,
                            output_scores=True,
                            output_attentions=True,
                            )
            
            generation = dict_outputs.sequences[:, input_length:].cpu()
            for gen in generation:
                generations.append(gen)
            
            batch_scores = dict_outputs.scores
            batch_scores_softmax = [[] for _ in range(len(batch_scores[0]))]
            for ind1, logits in enumerate(batch_scores): 
                for ind2, seq_logits in enumerate(logits):
                    batch_scores_softmax[ind2].append(seq_logits.softmax(0)[generation[ind2][ind1]].cpu())
            
            all_scores_softmax.extend(batch_scores_softmax)
                
            layers_to_process = args.layers
            hidden_states = dict_outputs.hidden_states
            attentions = dict_outputs.attentions
            ###### hidden_states : (num_tokens, num_layers, [num_seq, num_input_tokens/1, embedding_size])
            ###### attentions: (num_tokens, num_layers, [num_seq, num_heads, seq_len, seq_len])

            context_length = attentions[0][0].shape[-1]
            new_token_length = len(attentions)
            num_layers = len(attentions[0])
            num_heads = attentions[0][0].shape[1]

            # for ind in range(attentions[0][0].shape[0]):
            #     lookback_ratio = torch.zeros((num_layers, num_heads, new_token_length))
            #     for i in range(len(attentions)): # iterating over the new tokens length
            #         for l in range(num_layers):
            #             att_max_on_context = attentions[i][l][0, :, -1, :context_length].max(-1)
            #             att_max_all = attentions[i][l][0, :, -1, :].max(-1)
            #             attn_on_context = attentions[i][l][0, :, -1, :context_length].mean(-1)
            #             attn_on_new_tokens = attentions[i][l][0, :, -1, context_length:].mean(-1)
            #             lookback_ratio[l, :, i] = attn_on_context / (attn_on_context + attn_on_new_tokens)

            # for layer in layers_to_process:
            #     all_token_attentions_layer = {}
            #     for ind in range(attentions[1][-1].shape[0]):
            #         all_token_attentions_layer[ind + off_set] = []
            #         for attention in attentions:
            #             all_token_attentions_layer[ind + off_set].append(attention[layer][ind, -1, :].detach().cpu().float().numpy())

            clean_generations_range = []
            for generated_ids in generation:
                gen = tokenizer.decode(generated_ids, skip_special_tokens=True)
                clean_generation_decoded = dataset_egc(batch, gen, args.language)
                start_ind, end_ind = getCleanGenerationRange(generated_ids.tolist(), clean_generation_decoded, tokenizer)
                if start_ind is None or end_ind is None:
                    has_error = True
                    # print("gen:", gen)
                    # print("clean_generation_decoded:", clean_generation_decoded)
                    print(f'Cannot find clean generation range for {task_id_path}')
                    start_ind, end_ind = getGenerationRange(generated_ids.tolist(), tokenizer)
                    clean_generations_range.append((start_ind, end_ind, has_error))
                else:
                    has_error = False
                    clean_generations_range.append((start_ind, end_ind, has_error))

            for layer in layers_to_process:
                all_token_hidden_states_layer = {}
                for ind in range(hidden_states[1][-1].shape[0]):
                    all_token_hidden_states_layer[ind + off_set] = []
                    start_code_ind, end_code_ind, has_error = clean_generations_range[ind]
                    new_token_code_length = end_code_ind - start_code_ind
                    att_max_on_context = torch.zeros((new_token_code_length))
                    att_max_all = torch.zeros((new_token_code_length))
                    lookback_ratio = torch.zeros((new_token_code_length))
                    lookback_ratio_paper = torch.zeros((num_heads, new_token_length))
                    for i in range(new_token_code_length):
                        att_max_on_context[i] = attentions[start_code_ind + i][layer - 1][ind, :, -1, :context_length].max()
                        att_max_all[i] = attentions[start_code_ind + i][layer - 1][ind, :, -1, :].max()
                        att_on_context = attentions[start_code_ind + i][layer - 1][ind, :, -1, :context_length].mean()
                        att_on_new_tokens = attentions[start_code_ind + i][layer - 1][ind, :, -1, context_length:].mean()
                        lookback_ratio_heads = att_on_context / (att_on_context + att_on_new_tokens)
                        lookback_ratio[i] = lookback_ratio_heads.max()
                    for i in range(new_token_length):
                        att_on_context = attentions[i][layer - 1][ind, :, -1, :context_length].mean()
                        att_on_new_tokens = attentions[i][layer - 1][ind, :, -1, context_length:].mean()
                        lookback_ratio_paper[:, i] = att_on_context / (att_on_context + att_on_new_tokens)
                    att_max_on_context_max_token = int(att_max_on_context.argmax()) + start_code_ind
                    att_max_all_max_token = int(att_max_all.argmax()) + start_code_ind
                    lookback_ratio_max_token = int(lookback_ratio.argmax()) + start_code_ind
                    att_max_on_context_min_token = int(att_max_on_context.argmin()) + start_code_ind
                    att_max_all_min_token = int(att_max_all.argmin()) + start_code_ind
                    lookback_ratio_min_token = int(lookback_ratio.argmin()) + start_code_ind

                    print(new_token_length, start_code_ind, end_code_ind)
                    cc = {
                         "att_max_on_context_max_token": att_max_on_context_max_token,
                        "att_max_all_max_token": att_max_all_max_token,
                        "lookback_ratio_max_token": lookback_ratio_max_token,
                        "att_max_on_context_min_token": att_max_on_context_min_token,
                        "att_max_all_min_token": att_max_all_min_token,
                        "lookback_ratio_min_token": lookback_ratio_min_token,
                    }
                    print(cc)


                    all_token_hidden_states_layer[ind + off_set].append({
                        "has_error": has_error,
                        "att_max_on_context_max_token": att_max_on_context_max_token,
                        "att_max_all_max_token": att_max_all_max_token,
                        "lookback_ratio_max_token": lookback_ratio_max_token,
                        "att_max_on_context_min_token": att_max_on_context_min_token,
                        "att_max_all_min_token": att_max_all_min_token,
                        "lookback_ratio_min_token": lookback_ratio_min_token,
                        "lookback_ratio_paper": lookback_ratio_paper.detach().cpu().float().numpy(),
                        "hidden_states_att_max_on_context_max_token": hidden_states[att_max_on_context_max_token][layer][ind, -1, :].detach().cpu().float().numpy(),
                        "hidden_states_att_max_all_max_token": hidden_states[att_max_all_max_token][layer][ind, -1, :].detach().cpu().float().numpy(),
                        "hidden_states_lookback_ratio_max_token": hidden_states[lookback_ratio_max_token][layer][ind, -1, :].detach().cpu().float().numpy(),
                        "hidden_states_att_max_on_context_min_token": hidden_states[att_max_on_context_min_token][layer][ind, -1, :].detach().cpu().float().numpy(),
                        "hidden_states_att_max_all_min_token": hidden_states[att_max_all_min_token][layer][ind, -1, :].detach().cpu().float().numpy(),
                        "hidden_states_lookback_ratio_min_token": hidden_states[lookback_ratio_min_token][layer][ind, -1, :].detach().cpu().float().numpy(),
                    })

                    # for hidden_state, attention in zip(hidden_states, attentions):
                    #     all_token_hidden_states_layer[ind + off_set].append(hidden_state[layer][ind, -1, :].detach().cpu().float().numpy())

                if layer not in all_token_hidden_states_layer_list:
                    all_token_hidden_states_layer_list[layer] = {}
                all_token_hidden_states_layer_list[layer].update(all_token_hidden_states_layer)
            # return hidden_state
                
            del dict_outputs
            gc.collect()
            torch.cuda.empty_cache()
            layers = args.layers
            del hidden_states
            gc.collect()
            torch.cuda.empty_cache()
            num_gens -= len(generation)
            off_set += len(generation)
        
        for gen_ids in generations:
            generations_decoded.append(tokenizer.decode(gen_ids, skip_special_tokens=True))
        for layer in layers_to_process:
            layer_embeddings = all_token_hidden_states_layer_list[layer]
            layer_embeddings_dict = dict(
                    id=batch['task_id'][0],
                    layer_embeddings = layer_embeddings,
                )
            
            # print(f'Writing {len(sequences[layer])} generations to {cache_dir}...')
            pd.to_pickle(layer_embeddings_dict, os.path.join(cache_dir, f'all_att_chosen_token_embedding_{task_id_path}_{layer}.pkl'))
        generation_sequences_output = dict(
                prompt=tokenizer.decode(input_ids.cpu()[0], skip_special_tokens=True),
                id=batch['task_id'][0],
                problem=batch['original_prompt'][0],
                generations=generations_decoded,
                generations_ids=generations,
                softmax_scores=all_scores_softmax,
            )
        pd.to_pickle(generation_sequences_output, os.path.join(cache_dir, f'generation_sequences_output_{task_id_path}.pkl'))
        torch.cuda.empty_cache()
    
    
    # pd.to_pickle(generation_sequences_output, os.path.join(cache_dir, f'generation_sequences_output.pkl'))
    return

def main(overwrite=False, continue_from=None, parallel:int=None):
    time_start = time.time()
    if continue_from:
        fname = os.path.basename(continue_from)
        args.__dict__ = utils.jload(continue_from.replace(fname, 'args'+fname.replace("_partial.pkl", ".json")))
        old_sequences = pd.read_pickle(continue_from)
        cache_dir = os.path.dirname(continue_from)
        run_id = int(os.path.basename(continue_from).replace("_partial.pkl", ""))
        model_name = args.model
    else:
        old_sequences = []
        model_name = args.model
        cache_dir = OUTPUT_DIR
        old_results = glob.glob(os.path.join(OUTPUT_DIR, '*.pkl'))
        old_results = [_ for _ in old_results if '_partial' not in _]
        run_id = len(old_results)
        with open(os.path.join(OUTPUT_DIR, f'args{run_id}.json'), 'w') as f:
            json.dump(args.__dict__, f)
    print(f'Generating {args.num_generations_per_prompt} generations per prompt for {model_name} on {args.dataset}...')
    print(f"Saving to {os.path.join(cache_dir, f'{run_id}.pkl')}")
    temp_dir = os.path.join(cache_dir,'temp2')
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    get_generations(model_name, args, seed=args.seed, old_sequences=old_sequences,cache_dir=temp_dir)
    print("Total time: ", time.time() - time_start)
    print("Total time: ", time.time() - time_start, file=logInfo)
    return

if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    task_runner = main(parallel=args.nprocess)
