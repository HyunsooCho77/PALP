import os
import json
import logging
import torch
from datasets import Dataset, load_metric, DatasetDict, load_dataset
from tqdm import tqdm
import pickle
import pdb
import numpy as np
from typing import List, Tuple


def return_filenames(args, task_name):
    backbone_dict = {'EleutherAI/gpt-j-6B':'GPTJ', 'gpt2-large':'gpt2-large'}
    backbone_name = backbone_dict[args.model_name_or_path]
    
    output_dir = os.path.join(args.output_dir, backbone_name, task_name,f'{args.shots}_shots')
    os.makedirs(output_dir, exist_ok=True)

    if args.apply_prompt == False:
        train_fname='baseline_train.pickle' if args.shots == 'full' else f'baseline_train_seed_{args.seed}.pickle'
        test_fname='baseline_test.pickle' if args.shots == 'full' else f'baseline_test_seed_{args.seed}.pickle'
    else:
        if args.demonstrations == False:
            train_fname='PALP-T_train.pickle' if args.shots == 'full' else f'PALP-T_train_seed_{args.seed}.pickle'
            test_fname='PALP-T_test.pickle' if args.shots == 'full' else f'PALP-T_test_seed_{args.seed}.pickle'
        else:
            # if os.path.exists(os.path.join(output_dir, f'PALP-T_train_seed_{args.seed}.pickle')) and os.path.exists(os.path.join(output_dir, 'PALP-T_test_seed_{args.seed}.pickle')):
            train_fname='PALP-D_train.pickle' if args.shots == 'full' else f'PALP-D_train_seed_{args.seed}.pickle'
            test_fname='PALP-D_test.pickle' if args.shots == 'full' else f'PALP-D_test_seed_{args.seed}.pickle'
            # else:
            #     train_fname='PALP-T_train.pickle' if args.shots == 'full' else f'PALP-T_train_seed_{args.seed}.pickle'
            #     test_fname='PALP-T_test.pickle' if args.shots == 'full' else f'PALP-T_test_seed_{args.seed}.pickle'

    full_train_fname = os.path.join(output_dir,train_fname)
    full_test_fname = os.path.join(output_dir, test_fname)

    return full_train_fname, full_test_fname, output_dir


def extract_embedding(args, data_loader, model, num_labels, is_train):

    embeddings = [[] for _ in range(num_labels)] if is_train == True else []

    with torch.no_grad():
        for step, samples in tqdm(enumerate(data_loader), desc="Save Embeddings"):

            samples = {k: v.cuda() for k, v in samples.items()}
            labels = samples['labels'].cpu().tolist()

            del samples['labels']

            last_index = torch.ne(samples['attention_mask'], 0).sum(-1) - 1
            outputs = model(**samples)
            
            last_hidden_states = outputs.last_hidden_state.cpu()
            last_representation = last_hidden_states[range(last_hidden_states.size()[0]), last_index, :]            
            
            for batch_index, label in enumerate(labels):
                current_last_representation = last_representation[batch_index].cpu().tolist()
                if is_train == True:
                    embeddings[label].append(current_last_representation)
                else : 
                    embeddings.append([current_last_representation, label])

        
        return embeddings

def custom_load_dataset(task_name, benchmark_name):
    if task_name is not None and benchmark_name is not None:
        if benchmark_name == 'custom':
        # MR, CR
        # TODO : customize?
            train_path = f'./datasets/{task_name}/train.jsonl'
            test_path = f'./datasets/{task_name}/test.jsonl'

            train_dataset = load_dataset('json', data_files=train_path)['train']
            eval_dataset = load_dataset('json', data_files=test_path)['train']
        # SST-5, TREC, AGNews
        elif benchmark_name in ['glue', 'super_glue']:
            # glue, super_glue benchmarks
            train_dataset = load_dataset(benchmark_name, task_name, split=f'train')
            if task_name == 'mnli':
                # TODO : validation mismatched?
                eval_dataset = load_dataset(benchmark_name, task_name, split=f'validation_matched')
            else:
                eval_dataset = load_dataset(benchmark_name, task_name, split=f'validation')
        elif benchmark_name == 'huggingface':
            train_dataset = load_dataset(task_name, split='train')
            eval_dataset = load_dataset(task_name, split='test')
        else:
            # tweet_eval, clinc

            # FOR CLINC Dset
            # remove samples with class label oos (index 42)
            if benchmark_name == 'clinc_oos':
                train_dataset = load_dataset('clinc_oos', 'plus', split=f'train')
                eval_dataset = load_dataset('clinc_oos', 'plus', split=f'test')

                # num samples : 15250 -> 15000
                train_dataset = train_dataset.filter(lambda sample : sample['intent'] != 42)
                # set samples with label 150 to label 42
                train_dataset = train_dataset.map(lambda sample : {'intent' : 42} if sample['intent'] == 150 else {'intent' : sample['intent']})
                # num samples : 5500 -> 4500
                eval_dataset = eval_dataset.filter(lambda sample : sample['intent'] != 42)
                # set samples with label 150 to label 42
                eval_dataset = eval_dataset.map(lambda sample : {'intent' : 42} if sample['intent'] == 150 else {'intent' : sample['intent']})
            else:
                train_dataset = load_dataset(benchmark_name, task_name, split='train')
                eval_dataset = load_dataset(benchmark_name, task_name, split='test')

    else:
        raise NotImplementedError(f'{task_name} task is not implemented yet.')
    
    return train_dataset, eval_dataset





def logging_utils(args, logger):
    # Make one log on every process with the configuration for debugging.
    # Setup logging, we only want one process per machine to log things on the screen.
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
    logger.setLevel(logging.INFO)
    logging_output_file = os.path.join(args.output_dir, "output.log")
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    file_handler = logging.FileHandler(logging_output_file)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)


def train_converter(dataset, return_type = 'np_array'):

    class_num = len(dataset)

    rep_list = []
    label_list = []
    
    for class_idx in range(class_num):
        
        for k_shot, rep in enumerate(dataset[class_idx]):
            rep_list.append(rep)
            label_list.append(class_idx)
    
    if return_type == 'np_array':
        converted_dset = (np.array(rep_list), np.array(label_list))
        return converted_dset
    elif return_type == 'list':
        return rep_list, label_list

def test_converter(dataset, return_type = 'np_array'):

    rep_list = []
    label_list = []
    
    for rep,label in dataset:
        rep_list.append(rep)
        label_list.append(label)
    

    if return_type == 'np_array':
        converted_dset = (np.array(rep_list), np.array(label_list))
        return converted_dset
    elif return_type == 'list':
        return rep_list, label_list



