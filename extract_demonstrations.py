
import logging
import os
import pickle
import hydra
from omegaconf import DictConfig
from utils import custom_load_dataset, return_filenames, train_converter
from datasets import load_dataset
from dataset_utils import task_to_keys, task_templates
import numpy as np
from dataset_utils import task_to_keys
from typing import List, Tuple
from GDA import get_centroid
logger = logging.getLogger(__name__)

class TextRetrieval:
    def __init__(self, args, task_name, benchmark_name, reps: list, labels: list, output_dir: str):
        self.args = args
        self.reps = reps
        self.labels = labels
        self.task_name = task_name
        self.benchmark_name = benchmark_name
        self.texts, self.text_labels = self.load_text_data()
        self.save_path = os.path.join(output_dir, f"demonstrations")
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def load_text_data(self) -> Tuple[list, list]:
        if self.args.shots =='full':
            train_dataset, _ = custom_load_dataset(self.task_name, self.benchmark_name)
        else:
            train_dataset = load_dataset('json', data_files=os.path.join(self.args.few_dset_dir,f'{self.benchmark_name}-{self.task_name}-{self.args.shots}-{self.args.seed}','train.jsonl'))['train']
        
        if len(train_dataset) >= 30000:
            # ToDo: shuffle.....?
            dataset_split = train_dataset.train_test_split(train_size=30000, shuffle=True, seed=42)
            train_dataset = dataset_split['train']
        sentence1_key, sentence2_key = task_to_keys[self.task_name]['input']
        label_key = task_to_keys[self.task_name]['label']
        texts, text_labels = train_dataset[sentence1_key], train_dataset[label_key]

        # since representations and labels are in class sorted order, (0,...,0,1,...)
        ret_texts, ret_labels = [], []

        for i in range(len(set(text_labels))):
            for j in range(len(texts)):
                l = text_labels[j]
                if i == l:
                    ret_texts.append(texts[j])
                    ret_labels.append(text_labels[j])
        
        assert len(ret_texts) == len(texts)

        return ret_texts, ret_labels

    def rep_to_text(self, target_reps: np.array) -> dict:

        n_targets = target_reps.shape[0]
        text_list = []
        label_list = []
        text_idx = []

        for i in range(n_targets):
            target = target_reps[i].reshape(1, target_reps.shape[-1])
            t_diff = np.array(self.reps) - target
            t_idx = np.where(np.sum(t_diff, 1) == 0)[0][0]
            text_list.append(self.texts[t_idx])
            text_idx.append(t_idx)
            label_list.append(self.text_labels[t_idx])

        return {'text_idx': text_idx, 'texts':text_list, 'labels': label_list}

    def save_texts(self, text_dict: dict, file_name: str):
        text_list = text_dict['texts']
        demon_dict = {}

        task_dict = task_templates[self.task_name]
        for i in range(len(text_list)):
            demon_dict[i]= task_dict['prefix'] + text_list[i] + task_dict['postfix'] + task_dict['verbalizer'][i]

        with open(os.path.join(self.save_path, f"seed_{self.args.seed}.pickle"), 'wb') as fp:
            pickle.dump(demon_dict, fp)


def extract_text4mu(reps, class_true_mean, text_ret, data_name):
    '''extracting text that is closest to the class mean from labeled dataset'''
    closest_idx, c_dist = kmeans_assignment(np.array(reps), class_true_mean)
    text_dict = text_ret.rep_to_text(np.array(reps)[closest_idx])
    # assert len(text_dict['texts']) == n_class
    text_ret.save_texts(text_dict, f"")


def kmeans_assignment(centroids, points):

    num_centroids, dim = centroids.shape
    num_points, _ = points.shape

    # Tile and reshape both arrays into `[num_points, num_centroids, dim]`.                                                                      
    centroids = np.tile(centroids, [num_points, 1]).reshape([num_points, num_centroids, dim])
    points = np.tile(points, [1, num_centroids]).reshape([num_points, num_centroids, dim])

    # Compute all distances (for all points and all centroids) at once and                                                                       
    # select the min centroid for each point.                                                                                                    
    distances = np.sum(np.square(centroids - points), axis=2)
    return np.argmin(distances, axis=1), distances


# @hydra.main(config_path=".", config_name="model_config.yaml")
# def main(args: DictConfig) -> None:
def main(args):
    task_dic = dict(args.extract_tasks)

    for task_name, benchmark_name  in task_dic.items():
        train_file, test_file, output_dir = return_filenames(args, task_name)
        
        if os.path.exists(os.path.join(output_dir, f"demonstrations", f"seed_{args.seed}.pickle")):
            break
        # with open(os.path.join(test_file), 'rb') as f:
        #     rep_label = pickle.load(f)
        with open(os.path.join(train_file), 'rb') as f:
            aug_reps = pickle.load(f)
        reps, labels = train_converter(aug_reps, return_type='list')
        text_ret = TextRetrieval(args=args, task_name= task_name, benchmark_name = benchmark_name, reps=reps, labels=labels, output_dir=output_dir)
        
        class_true_mean = get_centroid(aug_reps)
        class_true_mean = class_true_mean.clone().detach().cpu().numpy()

        extract_text4mu(reps=reps, class_true_mean=class_true_mean, text_ret=text_ret, data_name=task_name)

        

if __name__ == "__main__":
    main()

