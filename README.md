# Prompt-Augmented Linear Probing: Scaling Beyond The Limit of Few-shot In-Context LearnersPrompt-Augmented Linear Probing
Official code for Prompt-Augmented Linear Probing (PALP). (AAAI 2023)
Paper: [Prompt-Augmented Linear Probing: Scaling Beyond The Limit of Few-shot In-Context Learners](https://arxiv.org/abs/2212.10873)

## Install following package
hydra version 1.1.1.
```bash
>> pip install hydra-core==1.1.1. 
```

## Usage.
Related hyper-parameters (tasks, models, etc) are in model_config.yaml

* Baseline: set apply_prompt=False, and demonstrations=False.
* PALP-T: set apply_prompt=True, and demonstrations=False.
* PALP-D: set apply_prompt=True, and demonstrations=True.

And run main code.

```bash
>> python run_PALP.py
```
