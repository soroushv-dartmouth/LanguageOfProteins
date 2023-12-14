# Functionalities:

## Pre-Training
### Tokenizer Training
The tokenizer for PROSSL is trained using the script 'Pre-Training/train_tokenizer.py'. This script requires setting the 'files' variable to the path of your data file. The data file should be a text file with each sequence separated by a newline.

### Model Pre-Training via MLM
The script 'Pre-Training/pretrain.py' is responsible for the pre-training of PROSSL. This step utilizes the same format for the data file as in tokenizer training. The tokenizer, trained in the previous step, is then loaded into the 'tokenizer' variable in this script.

## Master File Generation

The script 'master_initializer.py' located in the 'Data' directory is used to create master files in CSV format. These files include the ID, primary sequence, secondary sequence, and label for a set of proteins. For tasks like terminase and portal, pre-generated master files are available in 'Data/Master_Files'. To generate master files for other tasks, place primary structure FASTA files in 'Data/Primary_Structure_Files', and secondary structure files in 'Data/Secondary_Structure_Files/SS3_Files'. Modify the 'master_initializer.py' header to specify the task name.

## Train-Test Splitting

The script 'train_test_splitter_random.py' in 'Model_Fine-Tuning_and_Evaluation/Train-Test_Splitter' creates an 80/20 train-test split for certain task-specific data, as defined in the script's header. 'train_test_splitter_indexed.py' in the same directory performs a split where negative class instances are randomly divided (80/20), while positive class instances (like terminase, portal) are split as per user-defined criteria. For instance, the 'Portal_Split_IDs' and 'Terminase_Split_IDs' directories contain CSV files for various user-defined splits for the positive class. After running both scripts, a 'Splits' directory will be created, containing training and testing data for each split.

## Model Fine-Tuning and Evaluation

Both 'Primary_Structure_Task' and 'Secondary_Structure_Task' directories under 'Model_Fine-Tuning_and_Evaluation' contain a 'main.py' script. Once a train-test split is generated, these scripts can be used for fine-tuning, evaluating, and creating models using pre-trained models as a base. The script's termination results in a 'Outputs' directory, containing the fine-tuned model, a summary statistics text file, and a detailed CSV file of predictions and confidence scores for the test set. Fine-tuning hyperparameters are adjustable in the script headers.

## Model Cross-Validation

In the 'Model_Cross_Validation' directory, both the 'Primary_Structure_Task' and 'Secondary_Structure_Task' have a 'main.py' script for model fine-tuning, evaluation, and cross-validation (default set to 5-fold). The termination of these scripts creates an 'Outputs' directory with the fine-tuned model, a performance summary, and CSV files detailing predictions and confidence scores for each fold of the validation and test sets. Hyperparameters for fine-tuning are customizable in the script headers.

## Model Loading

Post fine-tuning, models can be tested on specific instances using the model loader feature in 'Model_Loader.' The 'Primary_Structure_Task' and 'Secondary_Structure_Task' directories each have a 'main.py' script. Edit these scripts to specify the path to the fine-tuned model and the data instances for testing.

## Confusion Matrix Generation

The 'make_confusion_matrices.py' script in the 'Statistics' directory requires a path to a directory of fine-tuned models. It generates a text file with confusion matrices for each task and split, reflecting the performance of the fine-tuned models.

## Unsupervised Evaluation

For unsupervised evaluations, place FASTA or text files of proteins of interest in 'Data/Proteins_of_Interest/Primary_Structure_Files' or 'Secondary_Structure_Files', depending on the task. The 'Primary_Structure_Task' and 'Secondary_Structure_Task' directories in 'Unsupervised_Evaluation' contain a 'main.py' script each. After setting up the files, specify the task in the script header to output unsupervised evaluation results, including various summary statistics.

## Fine-Tuned Models

Fine-tuned models for terminase and portal tasks are located in 'Fine-Tuned_Models', categorized by task and split type. Each model includes a performance summary text file and a detailed CSV file of predictions and scores. Confusion matrices for each task and split are found under 'Fine-Tuned_Models/Terminase_Task_Models' and 'Portal_Task_Models'. Additionally, cross-validation models are located under 'Terminase_CV_Models' and 'Portal_CV_Models'.
