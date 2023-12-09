import os
import glob
import pandas as pd
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# import
import copy
import torch
torch.cuda.is_available()
torch.cuda.empty_cache()

from transformers import RobertaConfig
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from transformers import RobertaForSequenceClassification
from transformers import AutoTokenizer
import shap
import numpy as np 
import scipy as sp
from transformers import PreTrainedTokenizerFast
from copy import deepcopy

# Function to predict label 
def get_prediction(text):
    max_length = 512

    # prepare our text into tokenized sequence
    inputs = tokenizer(text, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    # perform inference to our model
    outputs = tuned_model(**inputs)
    # get output probabilities by doing softmax
    probs = outputs[0].softmax(1)
    # executing argmax function to get the candidate label
    return probs.argmax()

# Function to make prediction on how "stereotypical" a sentence is 
def get_prediction_stereo(text):

    max_length = 512

    # prepare our text into tokenized sequence
    inputs = tokenizer(text, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    # perform inference to our model
    outputs = tuned_model(**inputs)

    outputs = tuned_model(**inputs)[0].detach().numpy()
    scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
    val = sp.special.logit(scores[:,1]) # use one vs rest logit units
    return val

# Batch prediction function for stereotypes
def f_batch_stereo(x):
    val = np.array([])
    for i in x:
    #   val = np.append(val, get_prediction(i))
      val = np.append(val, get_prediction_stereo(i))
    return val

def get_prediction_antistereo(text):

    max_length = 512

    # prepare our text into tokenized sequence
    inputs = tokenizer(text, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    # perform inference to our model
    outputs = tuned_model(**inputs)

    outputs = tuned_model(**inputs)[0].detach().numpy()
    scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
    val = sp.special.logit(scores[:,2]) # use one vs rest logit units
    return val

def f_batch_antistereo(x):
    val = np.array([])
    for i in x:
    #   val = np.append(val, get_prediction(i))
      val = np.append(val, get_prediction_antistereo(i))
    return val

things_to_explain = open('terminase_shap_seqs.txt', 'r').read().split('\n\n')
models = [var.split('\n')[0] for var in things_to_explain]
proteins_all = [var.split('\n')[1:] for var in things_to_explain]

pretrained_model_path = 'terminase_models/secondary/'
finetuned_model_path = 'secondary_finetuned_model.pt'

config = RobertaConfig.from_pretrained(pretrained_model_path)
config.num_labels = 2

model = RobertaForSequenceClassification.from_pretrained(pretrained_model_path, config=config)
tokenizer = PreTrainedTokenizerFast(tokenizer_file="{}byte-level-BPE.tokenizer.json".format(pretrained_model_path))
# tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# state_dict = torch.load(finetuned_model_path, map_location=torch.device('cpu'))
# model.load_state_dict(state_dict)

for i in range(len(models)):
    print('>' + models[i])
    print('\n')
    proteins = [var.split(' - ')[0] for var in proteins_all[i] if var != '']
    model_path = "terminase_models/secondary/{}".format(models[i])
    
    state_dict = torch.load(model_path+'/model.pt', map_location=torch.device('cpu'))
    tuned_model = deepcopy(model)
    tuned_model.load_state_dict(state_dict)
    # tuned_model = RobertaForSequenceClassification.from_pretrained(model_path)

    # data path
    paths = [str(x) for x in Path("Borrelia_garinii_txt").glob("*.txt")]

    # tokenizer 

    # tokenizer = PreTrainedTokenizerFast(tokenizer_file="byte-level-BPE.tokenizer.json")
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})



    # tuned_model = BertForSequenceClassification.from_pretrained("best-bert-base-uncased-0.5")
    # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

    

    explainer_stereo = shap.Explainer(f_batch_stereo, tokenizer)
    explainer_antistereo = shap.Explainer(f_batch_antistereo, tokenizer)

    exp_texts = []
    # print(proteins)
    # proteins_dict = open('non_portal_uniref_50_cluster_size_700_fixed.fasta', 'r').read().split('>')
    proteins_dict = open('terminase.fasta', 'r').read().split('>')
    del(proteins_dict[0])
    key_vals = [var.split('\n') for var in proteins_dict]
    proteins_keys = [var[0] for var in key_vals]
    proteins_vals = [''.join(var[1:]) for var in key_vals]
    # proteins_keys = [var[1:] for var in proteins_dict if var.startswith('>')]
    # proteins_vals = [var for var in proteins_dict if not var.startswith('>') and len(var) > 0]
    # print(len(proteins_keys), len(proteins_vals))
    assert(len(proteins_keys) == len(proteins_vals))
    proteins_dict = {proteins_keys[i]: proteins_vals[i] for i in range(len(proteins_keys))}
    revert_proteins_dict = {proteins_dict[var]:var for var in proteins_dict}
    prim_seqs = [proteins_dict[var] for var in proteins if var in proteins_dict]
    # print(prim_seqs)
    # print('-------------Lookup Started-------------')
    lookup_res = {}
    # ss3_dir = 'portal_non_portal_ss_data/non_portal_SS_40524'
    # ss3_dir = 'non_terminase_uniref_50_ss3'
    ss3_dir = 'terminase_ss3'
    ss3_files = glob.glob('{}/*.ss3'.format(ss3_dir))
    for ss3_file in ss3_files:
        df = pd.read_csv(ss3_file, delimiter='\t')
        prim = ''.join(df['AA'].to_list())
        if prim in prim_seqs:
            secondary = ''.join(df['SS'].to_list())
            # secondary = ''.join(df['AA'].to_list())
            lookup_res[revert_proteins_dict[prim]] = secondary
            continue

    # print('-------------Explain Started-------------')
    reverse_lookup_res = {lookup_res[var]: var for var in lookup_res}
    exp_texts = list(lookup_res.values())
    labels = [0]*len(exp_texts)
    test = {'label': labels, 'text': exp_texts}
    # print(test)
    shap_values = explainer_stereo(test, fixed_context=1)
    for i, shap_value in enumerate(shap_values): 
        print(reverse_lookup_res[exp_texts[i]])
        # print(shap_values[i].base_values)
        # print(get_prediction(exp_texts[i])) # get prediction from tuned model 
        shap_string = [] 
        for value, text in zip(shap_value.values, shap_value.data): # examine by-token value 
            shap_string.append(text + '(' + str(format(value, ".3f")) + ')')
        print(" ".join(shap_string))
