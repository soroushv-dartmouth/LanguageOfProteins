# Functionalities:

## Pre-Training:
### Training Tokenizer:
The script 'Pre-Training/train_tokenizer.py' handles the training of the tokenizer of PROSSL. The 'files' variable should be set to the path of the data file, a text file containing all the sequences separated by newline.

### Training Model via MLM:
The script 'Pre-Training/pretrain.py' handles the pre-training of PROSSL. The data file should be the same in format as that used to train the tokenizer. The trained tokenizer in the previous step is loaded into the variable 'tokenizer'.

## Master File Generation:

The executable 'master_initializer.py' in the 'Data' directory generates master files in csv format containing the ID, primary sequence, secondary sequence, and label for a given set of proteins. For the terminase and portal tasks, the master files are already pre-generated and located under 'Data/Master_Files'. For on-demand generation of master files for other tasks, place the desired primary structure FASTA files in 'Data/Primary_Structure_Files', place the desired secondary structure files(SS3 format) in a directory located under 'Data/Secondary_Structure_Files/SS3_Files', and edit the header of the executable 'master_initializer.py' to specify the name of the desired task.

## Train-Test Splitting:

The executable 'train_test_splitter_random.py' under 'Model_Fine-Tuning_and_Evaluation/Train-Test_Splitter' generates an 80/20 train-test split of some task-dependent data, as specified in the header of the code. The executable 'train_test_splitter_indexed.py', under the same directory, instead generates a data split where instances from the negative class(i.e.: nonterminase, nonportal) follow a random 80/20 train-test split, while instances from the positive class(i.e.: terminase, portal) are split according to the user's specifications. For instance, the directories 'Portal_Split_IDs' and 'Terminase_Split_IDs', both located under 'Model_Fine-Tuning_and_Evaluation/Train-Test_Splitter', each contain csv files of various user-specified train-test splits for instances of the positive class. Such splits are user-specified and can therefore be modified as needed. After running both executables(random and indexed splitters), a directory named 'Splits' will be generated, containing the training and testing data for each specified split.

## Model Fine-Tuning and Evaluation:

The directories 'Primary_Structure_Task' and 'Secondary_Structure_Task', located under 'Model_Fine-Tuning_and_Evaluation', both contain analogous 'main.py' executables. Once a train-test split has been generated, these executables can be run to fine-tune, evaluate, and produce models using the given pre-trained models as a starting point. After termination of either executable, it will produce a directory named 'Outputs', containing the saved fine-tuned model, alongside a text file containing some brief summary statistics on model performance, and a more detailed csv file containing a breakdown of the model's predictions and confidence scores on each instance of the given test set. The hyperparameters for fine-tuning can be found in the header of the executables, and can be adjusted according to user preference.

## Model Cross-Validation:

The directories 'Primary_Structure_Task' and 'Secondary_Structure_Task', located under 'Model_Cross_Validation', both contain analogous 'main.py' executables. These executables can be run to fine-tune, evaluate, and produce models with cross-validation using the given pre-trained models as a starting point. After termination of either executable, it will produce a directory named 'Outputs', containing the saved fine-tuned model, a text file containing some brief summary statistics on model performance, and csv files containing a breakdown of the model's predictions and confidence scores on each instance of the validation and test sets, for each cross-validation fold(executables are set to 5-fold cross-validation as a default). The hyperparameters for fine-tuning can be found in the header of the executables, and can be adjusted according to user preference.

## Model Loading:

After a fine-tuned model has been generated, it can be tested on specific instances using the model loader feature, located under 'Model_Loader'. The 'Primary_Structure_Task' and 'Secondary_Structure_Task' directories both contain analogous 'main.py' executables, and their respective headers can be edited in order specify the path to the fine-tuned model the user wishes to load. The executables can then be further edited by passing in the data instances the user wishes to test the fine-tuned model on.

## Confusion Matrix Generation:

Under the 'Statistics' directory, the executable 'make_confusion_matrices.py' takes in a user-specified path to a directory of fine-tuned models in its header, and outputs a text file containing the confusion matrices associated with the performance of the fine-tuned models for every specified task and split.

## Unsupervised Evaluation:

The directories 'Primary_Structure_Task' and 'Secondary_Structure_Task', located under 'Unsupervised_Evaluation', both contain analogous 'main.py' executables. In order to set up the necessary environment to perform unsupervised evaluations, FASTA or text files containing user-specified instances of data representing 'proteins of interest' should be placed within the directories 'Data/Proteins_of_Interest/Primary_Structure_Files' or 'Data/Proteins_of_Interest/Secondary_Structure_Files', depending on the user's choice of task. These directories already contain pre-generated files with certain proteins of interest, but are modifiable according to user demands. After the files containing the proteins of interest are set up, a task can be specified by the user in the header of either executable file. The program will then output the results of the unsupervised evaluations by displaying various summary statistics relating to the performance of the pre-trained model being evaluated.

## Fine-Tuned Models:

Fine-tuned models for both the terminase and portal tasks are available under 'Fine-Tuned_Models'. The fine-tuned models are arranged by task(primary structure and secondary structure), and by split(random split and user-specified splits for instances of the positive class). Each model is accompanied by a text file containing some brief summary statistics on model performance, and a more detailed csv file containing a breakdown of the model's predictions and confidence scores on each instance of the given test set. Additionally, a text file containing the confusion matrices associated with the performance of the fine-tuned models for every task and split can be found under 'Fine-Tuned_Models/Terminase_Task_Models' and 'Fine-Tuned_Models/Portal_Task_Models' for the terminase and portal tasks, respectively. Finally, cross-validation models for both tasks can be found under 'Fine-Tuned_Models/Terminase_CV_Models' and 'Fine-Tuned_Models/Portal_CV_Models' for the terminase and portal tasks, respectively.