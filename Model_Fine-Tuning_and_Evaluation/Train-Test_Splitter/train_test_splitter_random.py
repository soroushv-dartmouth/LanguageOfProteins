task_name = 'terminase'  # set to 'terminase' for terminase configuration. set to 'portal' for portal configuration.

import csv
import random
import os

for i in [0, 1]:

    master_path = ['../../Data/Master_Files/' + task_name + '_master.csv', '../../Data/Master_Files/non' + task_name + '_master.csv']

    csv_setting = ['w', 'a']
    write_header = [True, False]
    train_path = 'Splits/Random_Split/train.csv'
    test_path = 'Splits/Random_Split/test.csv'

    # Create the output directory if it doesn't exist
    output_directory = os.path.dirname(train_path)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Open the original csv file
    with open(master_path[i], 'r') as master:
        reader = csv.reader(master)

        # Create the output csv files
        with open(train_path, csv_setting[i], newline='') as train, \
                open(test_path, csv_setting[i], newline='') as test:

            # Create csv writers for the output files
            train_writer = csv.writer(train)
            test_writer = csv.writer(test)

            header = next(reader)

            if write_header[i]:

                train_writer.writerow(header)
                test_writer.writerow(header)

            # Iterate over each row in the input file
            for row in reader:
                # Randomly assign each row to either the training or testing file
                if random.random() < 0.8:
                    train_writer.writerow(row)
                else:
                    test_writer.writerow(row)
