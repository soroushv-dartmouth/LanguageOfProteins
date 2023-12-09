batch_size = 32
learning_rate = 5e-5
num_epochs = 10
max_len = 128

import os
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import RobertaForSequenceClassification, PreTrainedTokenizerFast, AdamW, \
    get_linear_schedule_with_warmup
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead, RobertaConfig
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

experiments = ['Random_Split', 'Split_1', 'Split_2', 'Split_3', 'Split_4', 'Split_5', 'Split_6', 'Split_7', 'Split_8']

for i in range(len(experiments)):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    torch.cuda.is_available()
    torch.cuda.empty_cache()

    # Initialize the lists
    X_train, X_test, y_train, y_test = [], [], [], []
    test_IDs = []

    # Open the training csv file and read its contents
    with open('../Train-Test_Splitter/Splits/' + experiments[i] + '/train.csv', 'r') as train_file:  # use list
        train_reader = csv.reader(train_file)
        next(train_reader)  # skip the header row
        for row in train_reader:
            X_train.append(row[2])  # SS column
            y_train.append(int(row[3]))  # Label column

    # Open the testing csv file and read its contents
    with open('../Train-Test_Splitter/Splits/' + experiments[i] + '/test.csv', 'r') as test_file:  # use list
        test_reader = csv.reader(test_file)
        next(test_reader)  # skip the header row
        for row in test_reader:
            test_IDs.append(row[0])  # ID column
            X_test.append(row[2])  # SS column
            y_test.append(int(row[3]))  # Label column

    model_path = '../../Pre-Trained_Models/Secondary_Structure_Model'

    config = RobertaConfig.from_pretrained(model_path)
    config.num_labels = 2

    model = RobertaForSequenceClassification.from_pretrained(model_path, config=config)
    model.classifier = RobertaClassificationHead(config)

    model.classifier.dense.weight.data.normal_(mean=0.0, std=config.initializer_range)
    model.classifier.out_proj.weight.data.normal_(mean=0.0, std=config.initializer_range)

    tokenizer = PreTrainedTokenizerFast(tokenizer_file="../../Pre-Trained_Models/byte-level-BPE.tokenizer.json")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    input_ids_training = tokenizer.batch_encode_plus(X_train, padding=True, truncation=True, max_length=max_len)['input_ids']
    attention_masks_training = []
    for seq in input_ids_training:
        seq_mask = [float(c>0) for c in seq]
        attention_masks_training.append(seq_mask)

    input_ids_testing = tokenizer.batch_encode_plus(X_test, padding=True, truncation=True, max_length=max_len)['input_ids']
    attention_masks_testing = []
    for seq in input_ids_testing:
        seq_mask = [float(c>0) for c in seq]
        attention_masks_testing.append(seq_mask)

    # Convert data to PyTorch tensors
    input_ids_training = torch.tensor(input_ids_training)
    input_ids_testing = torch.tensor(input_ids_testing)
    attention_masks_training = torch.tensor(attention_masks_training)
    attention_masks_testing = torch.tensor(attention_masks_testing)
    y_train = torch.tensor(y_train)
    y_test = torch.tensor(y_test)

    # Create DataLoader for training data
    train_data = TensorDataset(input_ids_training, attention_masks_training, y_train)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # Create DataLoader for testing data
    test_data = TensorDataset(input_ids_testing, attention_masks_testing, y_test)
    test_sampler = SequentialSampler(test_data)
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

        # Evaluate model on test data after each epoch
        model.eval()
        predictions, true_labels = [], []

        confidence_scores = []

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
            prob_positive = softmax_probs[:,1].flatten()

            predictions.extend(batch_predictions.cpu().numpy())
            true_labels.extend(batch_labels.cpu().numpy())

            confidence_scores.extend(prob_positive.cpu().numpy())

        # Compute accuracy
        accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, average='binary')
        auc = roc_auc_score(true_labels, predictions)


        print("Epoch:", epoch+1)
        print("Accuracy: {:.4f}".format(accuracy))
        print("F1: {:.4f}".format(f1))
        print("\n")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch_f1 = f1
            best_epoch_auc = auc
            best_epoch = epoch + 1
            best_model = model.state_dict()
            best_preds = predictions
            best_confidence_scores = confidence_scores


    print("Best Epoch:", best_epoch)
    print("Best Accuracy: {:.4f}".format(best_accuracy))
    print("Best Epoch F1: {:.4f}".format(best_epoch_f1))

    output_directory = 'Outputs/' + experiments[i]
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    torch.save(best_model, 'Outputs/' + experiments[i] + '/model.pt')

    data = list(zip(test_IDs, best_preds, best_confidence_scores, true_labels))

    # write the data to a csv file
    with open('Outputs/' + experiments[i] + '/predictions.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['ID', 'Prediction', 'Confidence Score', 'Label'])
        writer.writerows(data)

    with open('Outputs/' + experiments[i] + '/metrics.txt', 'w') as f:
        f.write(f'Accuracy: {best_accuracy:.4f}\n')
        f.write(f'F1 Score: {best_epoch_f1:.4f}\n')
        f.write(f'AUC: {best_epoch_auc:.4f}\n')
