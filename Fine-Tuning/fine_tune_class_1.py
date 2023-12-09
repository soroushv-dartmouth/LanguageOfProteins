# fine_tune_class_1.py
# fine tune the Robeta model2.1 for classification:

# set GPU
import os
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

# load previous model
model_path = "EsperBERTo"

config = RobertaConfig(
        vocab_size=3000,
        max_position_embeddings=514,
        num_attention_heads=12,
        num_hidden_layers=10,
        type_vocab_size=1,
        )

# load tokenizer
from transformers import RobertaTokenizerFast
tokenizer = RobertaTokenizerFast.from_pretrained(model_path, max_len=512)

# load model
from transformers import RobertaForMaskedLM
from transformers import RobertaForSequenceClassification

model1 = RobertaForMaskedLM.from_pretrained(model_path)
model2 = RobertaForSequenceClassification(config=config)
model2.roberta = copy.deepcopy(model1.roberta)
del model1
torch.cuda.empty_cache()

# path
paths = [str(x) for x in Path("/content/ss3-txt").glob("*.txt")]

# tokenizer 
tokenizer = ByteLevelBPETokenizer(model_path+"/vocab.json", model_path+"/merges.txt")

tokenizer._tokenizer.post_processor = BertProcessing(
        ("</s>", tokenizer.token_to_id("</s>")),
        ("<s>", tokenizer.token_to_id("<s>")),
        )

tokenizer.enable_truncation(max_length=512)

from transformers import PreTrainedTokenizerFast
tokenizer.save("byte-level-BPE.tokenizer.json")
tokenizer = PreTrainedTokenizerFast(tokenizer_file="byte-level-BPE.tokenizer.json")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# load dataset
from datasets import load_dataset
dataset_0 = load_dataset("text", data_dir="non_terminase_ss3_txt")
dataset_1 = load_dataset("text", data_dir="terminase_ss3_txt")

# add label
def add_label_0(d):
    d['labels'] = 0
    return d

dataset_0 = dataset_0.map(add_label_0)

def add_label_1(d):
    d['labels'] = 1
    return d

dataset_1 = dataset_1.map(add_label_1)

# build dataset
from datasets import concatenate_datasets
dataset = concatenate_datasets([dataset_0['train'],dataset_1['train']])

def tokenization(e):
    return tokenizer(e["text"], padding=True, truncation=True)

dataset = dataset.map(tokenization, batched=True, batch_size = len(dataset))

#def del_text(e):
#    del e['text']
#    return e

#dataset = dataset.map(del_text, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "labels"])

# fine-tune
from transformers import DataCollatorForTokenClassification
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
        output_dir="./Model_ft",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=16,
        save_steps=5_000,
        save_total_limit=10
        )

trainer = Trainer(
        model=model2,
        args=training_args,
        train_dataset=dataset,
        )

trainer.train()
trainer.save_model("./Model_ft")
