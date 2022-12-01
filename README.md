# GPT Distribution based classification

## Install following package
hydra version 1.1.1.
```bash
>> pip install hydra-core==1.1.1. 
```

## Performance report	
[Spreadsheet Link](https://docs.google.com/spreadsheets/d/1AgxKksixAC7LVdM7HGPq7wWmH5PZL5eW3y-5mY1e0ug/edit#gid=789844801)



## Extracting representations from GPTs
Related hyper-parameters (tasks, models, etc) are in model_config.yaml
```bash
>> python transformers_extract_embeddings.py
```


## Evaluation
Related hyper-parameters (tasks, models, etc) are in model_config.yaml
```bash
>> python metric_mahal.py
```

## Script (auto evaluating whole tasks)
Related hyper-parameters (tasks, models, etc) are in script_config.yaml
```bash
>> python script.py
```
