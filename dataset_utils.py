
#
# Utils for loading datasets from file (csv, tsv, ...).
# otherwise we use load_dataset() from huggingface library.
#

import csv
import random
import ast

def custom_generate_dataset_dict(filename):
    input_list = []
    label_list = []
    with open(filename) as f:
        validation_lines = csv.reader(f, delimiter='\t')
        # Remove header
        next(validation_lines, None)

        for validation_line in validation_lines:
            sample_index = validation_line[0]
            label = int(validation_line[1])
            input_sentence = validation_line[2]
            generation1 = validation_line[3]
            generation2 = validation_line[4]
            generation3 = validation_line[5]

            generation = '.'.join([generation1, generation2, generation3])

            input_sentence = generation + '.' + input_sentence

            label_list.append(label)
            input_list.append(input_sentence)
            
    return_dict = {
        'sentence' : input_list,
        'label' : label_list
    }

    return return_dict

task_to_keys = {
    # SINGLE SENTENCE TASKS
    # GLUE
    "cola" :  {
        'input' : ("sentence", None),
        'label' : 'label'
    },
    "sst2": {
        'input' : ("sentence", None),     # #labels = 2
        'label' : 'label',
    },
    "trec":  {
        'input' : ("text", None),         # #labels = 6
        'label' : 'coarse_label'
    },
    "trec2":  {
        'input' : ("text", None),         # #labels = 6
        'label' : 'fine_label'
    },
    # tweet_eval
    
    "stance_atheism" :  {
        'input' : ("text", None),
        'label' : 'label'
    },
    "emotion" :  {
        'input' : ("text", None),
        'label' : 'label'
    },
    "sentiment" :  {
        'input' : ("text", None),
        'label' : 'label'
    },
    "offensive" :  {
        'input' : ("text", None),
        'label' : 'label'
    },
    "rotten_tomatoes" :  {
        'input' : ("text", None),
        'label' : 'label'
    },
    "ag_news": {
        'input' : ("text", None),      # #labels = 4
        'label' : 'label',
    },
    "plus" :  {
        'input' : ("text", None),
        'label' : 'intent'
    },
    "banking77" :  {
        'input' : ("text", None),
        'label' : 'label'
    },
    
    # SENTENCE PAIR TASKS
    # glue
    "mnli":  {
        'input' : ("premise", "hypothesis"),
        'label' : 'label',
    },
    "mrpc":  {
        'input' : ("sentence1", "sentence2"),
        'label' : 'label',
    },
    "qqp":  {
        'input' : ("question1", "question2"),
        'label' : 'label',
    },
    "rte":  {
        'input' : ("sentence1", "sentence2"),
        'label' : 'label'
    },
    # super_glue
    "boolq" :  {
        'input' : ("question", "passage"),
        'label' : 'label'
    },
    "cb" :  {
        'input' : ("premise", "hypothesis"),
        'label' : 'label'
    },
}




task_templates={
    # SINGLE SENTENCE TASKS
    # GLUE
    "cola" :  {
        'prefix' : 'Sentence 1: ',
        'infix' : '',
        'postfix': '\nSentiment:',
        'verbalizer':{ 0 : " True", 1 : " False"}
    },
    "sst2": {
        'prefix' : 'Sentence 1: ',
        'infix' : '',
        'postfix': '\nSentiment:',
        'verbalizer':{ 0 : " negative", 1 : " positive"}
    },
    "trec":  {
        'prefix' : 'Sentence 1: ',
        'infix' : '',
        'postfix': '\nLabel:',
        'verbalizer':{  0 : " description", 1 : " entity", 2 : " expression",  3 : " human",  4 : " number",  5 : " location"}
    },
    # tweet_eval
    "stance_atheism" :  {
        'prefix' : 'Sentence 1: ',
        'infix' : '',
        'postfix': '\nLabel:',
        'verbalizer':{  0 : " none", 1 : " against", 2 : " favor"}
    },
    "emotion" :  {
        'prefix' : 'Sentence 1: ',
        'infix' : '',
        'postfix': '\nSentiment:',
        'verbalizer':{ 0 : " anger", 1 : " joy", 2 : " optimism", 3 : " sadness" }
    },
    "offensive" :  {
        'prefix' : 'Sentence 1: ',
        'infix' : '',
        'postfix': '\nSentiment:',
        'verbalizer':{  0 : " non-offensive", 1 : " offensive" }
    },
    "rotten_tomatoes" :  {
        'prefix' : 'Sentence 1: ',
        'infix' : '',
        'postfix': '\nSentiment:',
        'verbalizer':{ 0 : " negative", 1 : " positive"}
    },
    "ag_news": {
        'prefix' : 'Sentence 1: ',
        'infix' : '',
        'postfix': '\nLabel:',
        'verbalizer':{0 : " World", 1 : " Sports", 2 : " Business", 3 : " Technology"}
    },
    "plus" :  {
        'prefix' : 'Sentence 1: ',
        'infix' : '',
        'postfix': '\nLabel:',
        'verbalizer':{ }
    },
    "banking77" :  {
        'prefix' : 'Sentence 1: ',
        'infix' : '',
        'postfix': '\nLabel:',
        'verbalizer':{ }
    },

    # SENTENCE PAIR TASKS
    # glue
    "mnli":  {
        'prefix' : 'Sentence 1: ',
        'infix' : '\nSentence 2: ',
        'postfix': '\nLabel:',
        'verbalizer':{ 0 : " True", 1 : " Neither", 2 : " False"}
    },
    "mrpc":  {
        'prefix' : 'Sentence 1: ',
        'infix' : '\nSentence 2: ',
        'postfix': '\nLabel:',
        'verbalizer':{ 0 : " False", 1 : " True"}
    },
    "rte":  {
        'prefix' : 'Premise: ',
        'infix' : '\nHypothesis: ',
        'postfix': '\nLabel:',
        'verbalizer':{ 0 : " True", 1 : " False" }
    },
    # super_glue
    "boolq" :  {
        'prefix' : 'Premise: ',
        'infix' : '\nHypothesis: ',
        'postfix': '\nLabel:',
        'verbalizer':{ 0 : " False", 1 : " True" }
    },
    "cb" :  {
        'prefix' : 'Premise: ',
        'infix' : '\nHypothesis: ',
        'postfix': '\nLabel:',
        'verbalizer':{ 0 : " True", 1 : " False", 2 : " Neither"}
    },
    
}



task_verbalizer={




}
