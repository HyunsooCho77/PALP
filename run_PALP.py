from extract_embeddings import main as representation_extraction
from extract_demonstrations import main as extract_demonstration
from GDA import main as eval_GDA
from SLP import main as eval_SLP
from traditional_classification import main as eval_other_LPs
import hydra
from omegaconf import DictConfig

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

@hydra.main(config_path=".", config_name="model_config.yaml")
def main(args: DictConfig) -> None:

    seed_everything(args.seed)

    # baseline
    if args.apply_prompt == False:
        representation_extraction(args)
    # PALP-D
    elif args.demonstrations == True:
        args.demonstrations = False
        representation_extraction(args)
        extract_demonstration(args)
        args.demonstrations = True
        representation_extraction(args)
    # PALP-T
    else:
        representation_extraction(args)

    # test
    eval_other_LPs(args)
    eval_GDA(args)
    eval_SLP(args)

if __name__ == "__main__":
    main()