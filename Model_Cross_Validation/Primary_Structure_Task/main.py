task_name = 'terminase'  # set to 'terminase' for terminase configuration. set to 'portal' for portal configuration.

batch_size = 32
learning_rate = 5e-5
num_epochs = 10
max_len = 128

import os
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import RobertaForSequenceClassification, AdamW, \
    get_linear_schedule_with_warmup, AutoTokenizer
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead, RobertaConfig
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import random

# Load the data from the master files
data = []
with open('../../Data/Master_Files/' + task_name + '_master.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # skip the header row
    for row in reader:
        data.append(row)

with open('../../Data/Master_Files/non' + task_name + '_master.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # skip the header row
    for row in reader:
        data.append(row)

# Split the data into input sequences (X) and labels (y)
X, y, instance_ids = [], [], []
for row in data:
    instance_ids.append(row[0])  # Instance ID column
    X.append(row[2])  # SS column
    y.append(int(row[3]))  # Label column

# Split the data into positive and negative instances
positive_instances = [X[i] for i in range(len(X)) if y[i] == 1]
negative_instances = [X[i] for i in range(len(X)) if y[i] == 0]

# Randomly shuffle the positive and negative instances
random.seed(42)
random.shuffle(positive_instances)
random.shuffle(negative_instances)

# Select 10 positive and 10 negative instances for the test set
test_instances = positive_instances[:10] + negative_instances[:10]

# Remove the test instances from the main dataset
X_train_val = [X[i] for i in range(len(X)) if X[i] not in test_instances]
y_train_val = [y[i] for i in range(len(X)) if X[i] not in test_instances]
instance_ids_train_val = [instance_ids[i] for i in range(len(X)) if X[i] not in test_instances]

# Create the test dataset
X_test = test_instances
y_test = [1] * 10 + [0] * 10  # Labels for the test dataset (10 positive and 10 negative)
instance_ids_test = [instance_ids[i] for i in range(len(X)) if X[i] in test_instances]

# Perform 5-fold cross-validation with the remaining training data
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold = 1
for train_index, val_index in skf.split(X_train_val, y_train_val):
    print(f"Fold {fold}")

    # Split the data into training and validation sets for this fold
    X_train, X_val = [X_train_val[i] for i in train_index], [X_train_val[i] for i in val_index]
    y_train, y_val = [y_train_val[i] for i in train_index], [y_train_val[i] for i in val_index]
    instance_ids_train, instance_ids_val = [instance_ids_train_val[i] for i in train_index], [instance_ids_train_val[i] for i in val_index]

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    torch.cuda.is_available()
    torch.cuda.empty_cache()

    model_path = '../../Pre-Trained_Models/Primary_Structure_Model'

    config = RobertaConfig.from_pretrained(model_path)
    config.num_labels = 2

    model = RobertaForSequenceClassification.from_pretrained(model_path, config=config)
    model.classifier = RobertaClassificationHead(config)

    model.classifier.dense.weight.data.normal_(mean=0.0, std=config.initializer_range)
    model.classifier.out_proj.weight.data.normal_(mean=0.0, std=config.initializer_range)

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    input_ids_training = tokenizer.batch_encode_plus(X_train, padding=True, truncation=True, max_length=max_len)['input_ids']
    attention_masks_training = []
    for seq in input_ids_training:
        seq_mask = [float(c>0) for c in seq]
        attention_masks_training.append(seq_mask)

    input_ids_val = tokenizer.batch_encode_plus(X_val, padding=True, truncation=True, max_length=max_len)['input_ids']
    attention_masks_val = []
    for seq in input_ids_val:
        seq_mask = [float(c>0) for c in seq]
        attention_masks_val.append(seq_mask)

    input_ids_testing = tokenizer.batch_encode_plus(X_test, padding=True, truncation=True, max_length=max_len)['input_ids']
    attention_masks_testing = []
    for seq in input_ids_testing:
        seq_mask = [float(c>0) for c in seq]
        attention_masks_testing.append(seq_mask)

    # Convert data to PyTorch tensors
    input_ids_training = torch.tensor(input_ids_training)
    input_ids_val = torch.tensor(input_ids_val)
    input_ids_testing = torch.tensor(input_ids_testing)
    attention_masks_training = torch.tensor(attention_masks_training)
    attention_masks_val = torch.tensor(attention_masks_val)
    attention_masks_testing = torch.tensor(attention_masks_testing)
    y_train = torch.tensor(y_train)
    y_val = torch.tensor(y_val)
    y_test = torch.tensor(y_test)

    # Create DataLoader for training data
    train_data = TensorDataset(input_ids_training, attention_masks_training, y_train)
    train_sampler = torch.utils.data.RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # Create DataLoader for validation data
    val_data = TensorDataset(input_ids_val, attention_masks_val, y_val)
    val_sampler = torch.utils.data.SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

    # Create DataLoader for testing data
    test_data = TensorDataset(input_ids_testing, attention_masks_testing, y_test)
    test_sampler = torch.utils.data.SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    # Set up optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # Define loss function
    loss_fn = nn.CrossEntropyLoss()

    best_accuracy = 0

    # Fine-tune model
    for epoch in range(num_epochs):
        model.train()
        for batch in train_dataloader:
            batch_input_ids = batch[0].to(device)
            batch_attention_masks = batch[1].to(device)
            batch_labels = batch[2].to(device)
            optimizer.zero_grad()
            outputs = model(batch_input_ids, attention_mask=batch_attention_masks, labels=batch_labels)
            loss = outputs[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        # Evaluate model on validation data after each epoch
        model.eval()
        val_predictions, val_true_labels = [], []
        confidence_scores_val = []

        for batch in val_dataloader:
            batch_input_ids = batch[0].to(device)
            batch_attention_masks = batch[1].to(device)
            batch_labels = batch[2].to(device)
            with torch.no_grad():
                outputs = model(batch_input_ids, attention_mask=batch_attention_masks)
            logits = outputs.logits
            batch_predictions = torch.argmax(logits, dim=1).flatten()
            batch_labels = batch_labels.flatten()

            softmax = nn.Softmax(dim=1)
            softmax_probs = softmax(logits)
            prob_positive = softmax_probs[:, 1].flatten()

            val_predictions.extend(batch_predictions.cpu().numpy())
            val_true_labels.extend(batch_labels.cpu().numpy())

            confidence_scores_val.extend(prob_positive.cpu().numpy())

        # Compute accuracy and AUC on the validation set
        accuracy = accuracy_score(val_true_labels, val_predictions)
        auc_val = roc_auc_score(val_true_labels, confidence_scores_val)
        f1 = f1_score(val_true_labels, val_predictions, average='binary')

        print("Epoch:", epoch + 1)
        print("Validation Accuracy: {:.4f}".format(accuracy))
        print("Validation AUC: {:.4f}".format(auc_val))
        print("Validation F1: {:.4f}".format(f1))
        print("\n")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch_f1 = f1
            best_epoch_auc = auc_val
            best_epoch = epoch + 1
            best_model = model.state_dict()
            best_val_preds = val_predictions
            best_val_confidence_scores = confidence_scores_val

    # Evaluate the best model on the test set
    model.load_state_dict(best_model)
    model.eval()
    test_predictions, test_true_labels = [], []
    confidence_scores_test = []

    for batch in test_dataloader:
        batch_input_ids = batch[0].to(device)
        batch_attention_masks = batch[1].to(device)
        batch_labels = batch[2].to(device)
        with torch.no_grad():
            outputs = model(batch_input_ids, attention_mask=batch_attention_masks)
        logits = outputs.logits
        batch_predictions = torch.argmax(logits, dim=1).flatten()
        batch_labels = batch_labels.flatten()

        softmax = nn.Softmax(dim=1)
        softmax_probs = softmax(logits)
        prob_positive = softmax_probs[:, 1].flatten()

        test_predictions.extend(batch_predictions.cpu().numpy())
        test_true_labels.extend(batch_labels.cpu().numpy())

        confidence_scores_test.extend(prob_positive.cpu().numpy())

    # Compute accuracy and AUC on the test set
    test_accuracy = accuracy_score(test_true_labels, test_predictions)
    test_auc = roc_auc_score(test_true_labels, confidence_scores_test)
    test_f1 = f1_score(test_true_labels, test_predictions, average='binary')

    print("Best Epoch:", best_epoch)
    print("Best Validation Accuracy: {:.4f}".format(best_accuracy))
    print("Best Validation AUC: {:.4f}".format(best_epoch_auc))
    print("Best Validation F1: {:.4f}".format(best_epoch_f1))
    print("Test Accuracy: {:.4f}".format(test_accuracy))
    print("Test AUC: {:.4f}".format(test_auc))
    print("Test F1: {:.4f}".format(test_f1))
    print("\n")

    # Save the model and predictions for this fold
    output_dir = 'Outputs'
    fold_dir = os.path.join(output_dir, f'Fold_{fold}')
    os.makedirs(fold_dir, exist_ok=True)

    torch.save(best_model, os.path.join(fold_dir, 'model.pt'))

    val_data = list(zip(instance_ids_val, best_val_preds, best_val_confidence_scores, val_true_labels))
    with open(os.path.join(fold_dir, 'validation_predictions.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Instance ID', 'Prediction', 'Confidence Score', 'Label'])
        writer.writerows(val_data)

    test_data = list(zip(instance_ids_test, test_predictions, confidence_scores_test, test_true_labels))
    with open(os.path.join(fold_dir, 'test_predictions.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Instance ID', 'Prediction', 'Confidence Score', 'Label'])
        writer.writerows(test_data)

    with open(os.path.join(fold_dir, 'metrics.txt'), 'w') as f:
        f.write(f'Validation Accuracy: {best_accuracy:.4f}\n')
        f.write(f'Validation AUC: {best_epoch_auc:.4f}\n')
        f.write(f'Validation F1 Score: {best_epoch_f1:.4f}\n')
        f.write(f'Test Accuracy: {test_accuracy:.4f}\n')
        f.write(f'Test AUC: {test_auc:.4f}\n')
        f.write(f'Test F1 Score: {test_f1:.4f}\n')

    fold += 1
