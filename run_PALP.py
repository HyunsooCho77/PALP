from extract_embeddings import main as representation_extraction
from extract_demonstrations import main as extract_demonstration
from GDA import main as eval_GDA
from SLP import main as eval_SLP
from traditional_classification import main as eval_other_LPs
import hydra
from omegaconf import DictConfig



@hydra.main(config_path=".", config_name="model_config.yaml")
def main(args: DictConfig) -> None:
    
    if args.apply_prompt == False:
        representation_extraction()    
    elif args.demonstrations == True:
        args.demontrations = False
        representation_extraction()
        extract_demonstration()
        args.demontrations = True
        representation_extraction()

    eval_other_LPs()
    eval_GDA()
    eval_SLP()
