#
# train_model_1.py
#
# usage: python train_model_1.py
#
# author: Zechen Yang @JeffZC
# 
# mentor: Weicheng Ma; advisor: Prof. Vosoughi
# 
# cite: huggingface.co
#
# train model 1 (pretain)

# import
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
import numpy as np
import csv
from transformers import TrainerCallback

# model path
model_path = '/home/jeff/model/Model_1'

# tokenizer
'''
tokenizer = ByteLevelBPETokenizer(
    "EsperBERTo/vocab.json",
    "EsperBERTo/merges.txt",
)
'''

import torch
torch.cuda.is_available()

# roberta config
from transformers import RobertaConfig
config = RobertaConfig(
    vocab_size=3000,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=10,
    type_vocab_size=1,
)

# roberta tokenizer
from transformers import RobertaTokenizerFast
tokenizer = RobertaTokenizerFast.from_pretrained(model_path, max_len=512)

# roberta model
from transformers import RobertaForMaskedLM
model = RobertaForMaskedLM(config=config)

# dataset
from transformers import LineByLineTextDataset
dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="train_ds",
    block_size=128,
)

# data collator
from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.20
)

# train
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir=model_path,
    overwrite_output_dir=True,
    num_train_epochs=100,
    per_gpu_train_batch_size=128,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset
)


trainer.train()

trainer.save_model(model_path)
