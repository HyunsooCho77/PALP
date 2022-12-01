
import logging
import os
import copy
import random
import sys
import pickle
import pdb
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer,
    DataCollatorWithPadding
)
import torch
from utils import custom_load_dataset, extract_embedding, return_filenames
from datasets import load_dataset, concatenate_datasets
from dataset_utils import task_to_keys, task_templates
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# HJ : generate final demonstrations by concatenating all demonstrations
def generate_demonstration(ordering, demo_dict):
    prompt = None
    for index in ordering:
        demo = demo_dict.get(index)
        if prompt is None:
            prompt = demo
        else:
            prompt = prompt + '\n' + demo
    return prompt

# HJ
def generate_ordering_list(class_list, initial_odering, max_samples=10):
    ordering_list = [initial_odering]

    bar = tqdm(range(max_samples), desc=f'Select {max_samples} samples')
    while True:
        random.shuffle(class_list)
        ordering = copy.deepcopy(class_list)
        # print(class_list)
        if len(ordering_list) == max_samples:
            break
        if ordering not in ordering_list:
            bar.update(1)
            ordering_list.append(ordering)
       
    print(f'Done selecting {len(ordering_list)} samples.')
    return ordering_list



def preprocess_sentence(args, task_name, sentence1, sentence2, demonstrations=''):
    def punctuation(sentence):
        if sentence !='':
            if sentence[-1] != '.' and sentence[-1] !='?':
                sentence +='. '
            else:
                sentence +=' '
        return sentence
    
    sentence1, sentence2 = punctuation(sentence1), punctuation(sentence2)
    if args.apply_prompt == True:
        final_sentence = task_templates[task_name]['prefix'] + sentence1 + task_templates[task_name]['infix'] + sentence2 + task_templates[task_name]['postfix'] 
    
    if args.demonstrations == True:
        # To-Do 어케 붙이지?
        # demonstrations = ""
        final_sentence = demonstrations + '\n' + final_sentence

    return final_sentence


@hydra.main(config_path=".", config_name="model_config.yaml")
def main(args: DictConfig) -> None:

    # Load pretrained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    # For gpt-2
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    try:
        model = AutoModel.from_pretrained(
                    args.model_name_or_path, 
                    revision="float16",             # specific model version to use. We use FP16 model
                    torch_dtype=torch.float16,  
                    low_cpu_mem_usage=True,         # keep RAM usage to 1x
            )
        model = model.cuda()
    except:
        model = AutoModel.from_pretrained(
                args.model_name_or_path,
                torch_dtype=torch.bfloat16,  
        ).to('cuda')
    model.eval()


    # Trarget extraction tasks
    task_dic = dict(args.extract_tasks)

    for task_name, benchmark_name  in task_dic.items():
        
        full_train_fname, full_test_fname, output_dir = return_filenames(args, task_name)

        
        if os.path.exists(full_train_fname) and os.path.exists(full_test_fname):
            print(f'Previous representation exists. name: f{full_train_fname}')
            sys.exit(0)
        else:
            print(f'continue: task: {task_name}, benchmark: {benchmark_name}')

        if args.shots == 'full':
            train_dataset, eval_dataset = custom_load_dataset(task_name, benchmark_name)
        else:
            train_dataset = load_dataset('json', data_files=os.path.join(args.few_dset_dir,f'{benchmark_name}-{task_name}-{args.shots}-{args.seed}','train.jsonl'))['train']
            _, eval_dataset = custom_load_dataset(task_name, benchmark_name)

        if len(train_dataset) >= 30000:
            dataset_split = train_dataset.train_test_split(train_size=30000, shuffle=True, seed=42)
            train_dataset = dataset_split['train']
        
        sentence1_key, sentence2_key = task_to_keys[task_name]['input']
        label_key = task_to_keys[task_name]['label']




        DEMONSTRATION = ''

        # tokenize input
        def preprocess_function(examples):
            # Tokenize the texts
            texts = ((examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key]))

            results = dict()
            sample_num = len(texts[0])
            input_ids_list = []
            attention_mask_list = []
            for sample_index in range(sample_num):

                sentence1 = texts[0][sample_index]
                sentence2 = texts[1][sample_index] if sentence2_key else ''

                final_sentence = preprocess_sentence(args, task_name, sentence1, sentence2, DEMONSTRATION)

                if os.path.exists(os.path.join(output_dir, f'sample.txt')) == False:
                    with open(os.path.join(output_dir, f'sample.txt'),'w') as f:
                        f.write(final_sentence)

                if len(final_sentence) == 0:
                    logger.info(f'SKIP SAMPLE. INDEX {sample_index} : INPUT WITH LENGTH 0 > {texts[0][sample_index]}')
                    continue

                result = tokenizer(final_sentence, max_length=2048, truncation=True) # padding='max_length')
                input_ids = result['input_ids']
                attention_mask = result['attention_mask']

                input_ids_list.append(input_ids)
                attention_mask_list.append(attention_mask)

            results['input_ids'] = input_ids_list
            results['attention_mask'] = attention_mask_list
            results['label'] = examples[label_key]

            return results

        # HJ : generate demonstrations
        # TODO : move to config?
        NUM_MAX_DEMO = 20
        demon_list = []
        if args.demonstrations == True:
            # data path
            full_demon_fname = os.path.join(output_dir, 'demonstrations', f'seed_{args.seed}.pickle')

            with open(full_demon_fname, 'rb') as f:
                demon_dict = pickle.load(f)

            indices = sorted(list(demon_dict.keys()))
            initial_orderings = sorted(list(demon_dict.keys()))

            demo_orderings = generate_ordering_list(indices, initial_orderings, max_samples=NUM_MAX_DEMO)
            for demo_ordering in demo_orderings:
                demo = generate_demonstration(demo_ordering, demon_dict)
                demon_list.append(demo)

            print(f'Generated {len(demon_list)} demonstrations.')
            
            dataset_list = []
            for i, demonstration in enumerate(demon_list):
                DEMONSTRATION = demonstration
                # Loading tokenized dataset
                print(f'\n\nTrain dataset mapping WITH DEMONSTRATION INDEX {i} ...{len(train_dataset)}')
                tmp_dataset = train_dataset.map(preprocess_function, batched=True, desc='Tokenizing train set.', remove_columns=train_dataset.column_names)
                dataset_list.append(tmp_dataset)
            train_dataset = concatenate_datasets(dataset_list)
        else:
            # Loading tokenized dataset
            print(f'\n\nTrain dataset mapping WITHOUT DEMONSTRATION...{len(train_dataset)}')
            train_dataset = train_dataset.map(preprocess_function, batched=True, desc='Tokenizing train set.', remove_columns=train_dataset.column_names)
        
        # pdb.set_trace()

        print(f'\n\nEvaluation dataset mapping...{len(eval_dataset)}')

        if args.demonstrations == True:
            DEMONSTRATION = demon_list[0]
        else:
            DEMONSTRATION = ''
        eval_dataset = eval_dataset.map(preprocess_function, batched=True, desc='Tokenizing test set.', remove_columns=eval_dataset.column_names)
        num_labels = len(set(train_dataset['label']))

        print(f'TOTAL DATSET : TRAIN ({len(train_dataset)}), EVAL ({len(eval_dataset)})')

        # pdb.set_trace()
        # Loading datasets 
        data_collator = DataCollatorWithPadding(tokenizer, padding=True, max_length=2048) # padding=True, trauncation=True, max_length=2048)
        train_dataloader = DataLoader(train_dataset, shuffle=False, collate_fn=data_collator, batch_size=args.extraction_batch_size)
        eval_dataloader = DataLoader(eval_dataset, shuffle=False, collate_fn=data_collator, batch_size=args.extraction_batch_size)

        # Extracts & saves representations
        train_embeddings = extract_embedding(args, train_dataloader, model, num_labels, is_train = True)
        with open(full_train_fname, 'wb') as f:
            pickle.dump(train_embeddings, f, pickle.HIGHEST_PROTOCOL)
        test_embeddings = extract_embedding(args, eval_dataloader, model, num_labels, is_train = False)
        with open(full_test_fname, 'wb') as f:
            pickle.dump(test_embeddings, f, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()

