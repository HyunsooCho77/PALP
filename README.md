# Prompt-Augmented Linear Probing
Official code for PALP (AAAI 2023).

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
