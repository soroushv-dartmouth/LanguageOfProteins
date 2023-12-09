task_name = 'terminase'  # set to 'terminase' for terminase configuration. set to 'portal' for portal configuration.

import os
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from transformers import RobertaConfig, RobertaForSequenceClassification, AutoTokenizer

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

pretrained_model_path = '../../Pre-Trained_Models/Primary_Structure_Model'

config = RobertaConfig.from_pretrained(pretrained_model_path)
config.num_labels = 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = RobertaForSequenceClassification.from_pretrained(pretrained_model_path, config=config,
                                                         ignore_mismatched_sizes=True)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)

model.to(device)
model.eval()

# Loading protein sequences from files

def load_sequences(file_path):
    with open(file_path, 'r') as file:
        sequences = file.read().split('>')[1:]
        sequences = [sequence.split('\n', 1)[1] for sequence in sequences]
    return sequences

# Load protein IDs for the protein of interest
with open('../../Data/Proteins_of_Interest/Primary_Structure_Files/' + task_name + '.fasta', 'r') as file:
    protein_of_interest_lines = file.read().splitlines()
    protein_of_interest_ids = [line.split('>')[1] for line in protein_of_interest_lines[::2]]

for j in range(len(protein_of_interest_ids)):

    # Loading protein of interest ID and sequence
    protein_of_interest_id = protein_of_interest_ids[j]
    protein_of_interest_sequence = protein_of_interest_lines[1 + 2 * j]

    # Loading dataset sequences
    positive_sequences = load_sequences('../../Data/Primary_Structure_Files/' + task_name + '.fasta')
    negative_sequences = load_sequences('../../Data/Primary_Structure_Files/non' + task_name + '.fasta')

    # Encoding protein of interest sequence
    protein_of_interest_tokens = tokenizer(protein_of_interest_sequence, truncation=True, padding='max_length',
                                           max_length=128, return_tensors="pt")
    protein_of_interest_tokens = protein_of_interest_tokens.to(device)
    with torch.no_grad():
        digits = model(**protein_of_interest_tokens, output_hidden_states=True)
        protein_of_interest_encoding = digits[1][-1][0][0].view(-1).cpu().numpy()

    # Calculate distances based on sequence encodings
    positive_distances = []
    negative_distances = []

    # Calculate distances for positive sequences
    for sequence in positive_sequences:
        positive_tokens = tokenizer(sequence, truncation=True, padding='max_length', max_length=128,
                                     return_tensors="pt")
        positive_tokens = positive_tokens.to(device)
        with torch.no_grad():
            digits = model(**positive_tokens, output_hidden_states=True)
            sequence_encoding = digits[1][-1][0][0].view(-1).cpu().numpy()
            distance = np.dot(sequence_encoding, protein_of_interest_encoding) / (
                    np.linalg.norm(sequence_encoding) * np.linalg.norm(protein_of_interest_encoding)
            )
            positive_distances.append(distance)

    # Calculate distances for non-positive sequences
    for sequence in negative_sequences:
        negative_tokens = tokenizer(sequence, truncation=True, padding='max_length', max_length=128,
                                        return_tensors="pt")
        negative_tokens = negative_tokens.to(device)
        with torch.no_grad():
            digits = model(**negative_tokens, output_hidden_states=True)
            sequence_encoding = digits[1][-1][0][0].view(-1).cpu().numpy()
            distance = np.dot(sequence_encoding, protein_of_interest_encoding) / (
                    np.linalg.norm(sequence_encoding) * np.linalg.norm(protein_of_interest_encoding)
            )
            negative_distances.append(distance)

    # Combine distances
    total_distances = sorted(positive_distances + negative_distances)

    # Calculate statistics
    error = np.zeros(len(total_distances))
    TP = np.zeros(len(total_distances))
    FP = np.zeros(len(total_distances))
    TN = np.zeros(len(total_distances))
    AUC = 0

    for i in range(len(total_distances)):
        TP[i] = np.mean(np.array(positive_distances) > total_distances[i])
        FP[i] = np.mean(np.array(negative_distances) > total_distances[i])
        TN[i] = np.mean(np.array(negative_distances) < total_distances[i])
        if i > 0:
            AUC += (FP[i - 1] - FP[i]) * TP[i]
        FN = 1 - TP[i]
        error[i] = FP[i] + FN

    optimal = np.argmin(error)
    opt_sens = TP[optimal]
    opt_spec = TN[optimal]
    opt_fp = FP[optimal]

    # Calculate AUC
    y_true = np.concatenate([np.ones(len(positive_distances)), np.zeros(len(negative_distances))])
    y_scores = np.concatenate([positive_distances, negative_distances])
    auc = roc_auc_score(y_true, y_scores)

    # Print results
    print("Protein ID:", protein_of_interest_id)
    print("True Positive Rate:", opt_sens)
    print("True Negative Rate:", opt_spec)
    print("AUC:", auc)
