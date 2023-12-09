results_dir = "../Fine-Tuned_Models/Terminase_Task_Models" # set to desired directory path.

import os
import pandas as pd
from sklearn.metrics import confusion_matrix

# Define the task names and split names
tasks = ["Primary_Structure_Task", "Secondary_Structure_Task"]
splits = ["Random_Split", "Split_1", "Split_2", "Split_3", "Split_4", "Split_5", "Split_6", "Split_7", "Split_8"]

# Open the text file for writing
with open("confusion_matrices.txt", "w") as file:

    file.write(f"Confusion Matrix Format:\n")
    file.write("[[TN, FP],\n [FN, TP]]\n\n")

    # Iterate over each task and split
    for task in tasks:
        for split in splits:
            # Construct the path to the predictions.csv file
            csv_path = os.path.join(results_dir, task, split, "predictions.csv")

            # Load the CSV file as a DataFrame
            df = pd.read_csv(csv_path)

            # Extract the Prediction and Label columns
            predictions = df["Prediction"]
            labels = df["Label"]

            # Compute the confusion matrix
            cm = confusion_matrix(labels, predictions)

            # Write the confusion matrix to the file
            file.write(f"{task}, {split}:\n")
            file.write(str(cm))
            file.write("\n\n")
