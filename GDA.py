import os
import pickle
import torch
import numpy as np
from sklearn.covariance import *
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
from utils import return_filenames

import random
import torch
import numpy as np
import os

def seed_everything(seed=1234):
    print(f'SET RANDOM SEED = {seed}')
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_gaussian_dist(aug_reps):
    num_classes = len(aug_reps)
    num_dim = len(aug_reps[0][0]) 
    class_mean = torch.zeros(num_classes, num_dim).cuda()
    
    for idx, reps_list in enumerate(aug_reps):
        reps_list = torch.tensor(reps_list).cuda()
        class_mean[idx] = (reps_list.mean(0))

    centered_aug_reps = []
    for idx, reps_list in enumerate(aug_reps):
        reps_list = torch.tensor(reps_list).cuda()
        centered_aug_reps.append((reps_list - class_mean[idx]).clone().detach().cpu().numpy())
    centered_aug_reps = np.concatenate(centered_aug_reps)
    
    # print('Fitting Covariance Matrix...')
    precision = LedoitWolf().fit(centered_aug_reps).precision_.astype(np.float32)
    class_var = torch.from_numpy(precision).float().cuda()

    return class_mean, class_var

def get_centroid(aug_reps):
    num_classes = len(aug_reps)
    num_dim = len(aug_reps[0][0]) 
    class_mean = torch.zeros(num_classes, num_dim).cuda()
    
    for idx, reps_list in enumerate(aug_reps):
        reps_list = torch.tensor(reps_list).cuda()
        class_mean[idx] = (reps_list.mean(0))

    centered_aug_reps = []
    for idx, reps_list in enumerate(aug_reps):
        reps_list = torch.tensor(reps_list).cuda()
        centered_aug_reps.append((reps_list - class_mean[idx]).clone().detach().cpu().numpy())
    centered_aug_reps = np.concatenate(centered_aug_reps)

    return class_mean



def calculate_acc(train_rep, train_label, class_mean, class_var):

    maha_score = []
    for c in range(len(class_mean)):
        train_rep = torch.tensor(train_rep).cuda()
        # train_rep = F.normalize(train_rep, dim=-1)
        centered_rep = train_rep - class_mean[c].unsqueeze(0)
        md = torch.diag(centered_rep @ class_var @ centered_rep.t())
        maha_score.append(md)
        
    maha_score = torch.stack(maha_score)
    pred = maha_score.argmin().item()

    correct = 1 if train_label == pred else 0
    
    return correct




# @hydra.main(config_path=".", config_name="model_config.yaml")
# def main(args: DictConfig) -> None:
def main(args):    
    seed_everything(args.seed)
    task_name=args.task_name
    train_file, test_file, _ = return_filenames(args, task_name)

    with open(os.path.join(test_file), 'rb') as f:
        rep_label = pickle.load(f)
    with open(os.path.join(train_file), 'rb') as f:
        aug_reps = pickle.load(f)

    class_mean, class_var = get_gaussian_dist(aug_reps)
    
    # Calculate Accuracy
    correct, total = 0,0
    for (rep, label) in rep_label:
        correct += calculate_acc(rep, label, class_mean, class_var)
        total +=1
    acc = correct*100/total
    print(f'** GDA score : {acc:.2f}')
          
if __name__ == "__main__":
    main()