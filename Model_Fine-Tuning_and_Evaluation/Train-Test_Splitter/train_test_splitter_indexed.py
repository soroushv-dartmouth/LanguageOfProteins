task_name = 'terminase'  # set to 'terminase' for terminase configuration. set to 'portal' for portal configuration.

import csv
import random
import os


def indexed_split():
    for i in range(1, 9):

        master_path = '../../Data/Master_Files/' + task_name + '_master.csv'
        split_path = task_name.capitalize() + '_Split_IDs/split' + str(i) + '.csv'

        train_path = 'Splits/Split_' + str(i) + '/train.csv'
        test_path = 'Splits/Split_' + str(i) + '/test.csv'

        # Create the output directories if they don't exist
        for path in [train_path, test_path]:
            output_directory = os.path.dirname(path)
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)

        with open(master_path, 'r') as master:
            master_reader = csv.reader(master)

            with open(split_path, 'r') as split:
                split_reader = csv.reader(split)

                with open(train_path, 'w', newline='') as train:
                    train_writer = csv.writer(train)
                    train_writer.writerow(['ID', 'AA', 'SS', 'Label'])

                    with open(test_path, 'w', newline='') as test:
                        test_writer = csv.writer(test)
                        test_writer.writerow(['ID', 'AA', 'SS', 'Label'])

                        next(split_reader)

                        for line in split_reader:
                            for n in [0, 1]:
                                item = line[n]

                                if item == '':
                                    break

                                master.seek(0)
                                next(master_reader)

                                for row in master_reader:
                                    if '|' in row[0]:
                                        ID = row[0].split('|')[-2]
                                    else:
                                        ID = row[0]

                                    if item == ID:
                                        if n == 0:
                                            train_writer.writerow(row)
                                        else:
                                            test_writer.writerow(row)
                                        break


def random_split():
    for i in range(1, 9):

        master_path = '../../Data/Master_Files/non' + task_name + '_master.csv'

        train_path = 'Splits/Split_' + str(i) + '/train.csv'
        test_path = 'Splits/Split_' + str(i) + '/test.csv'

        # Create the output directories if they don't exist
        for path in [train_path, test_path]:
            output_directory = os.path.dirname(path)
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)

        # Open the original csv file
        with open(master_path, 'r') as master:
            reader = csv.reader(master)

            # Create the output csv files
            with open(train_path, 'a', newline='') as train, \
                    open(test_path, 'a', newline='') as test:

                # Create csv writers for the output files
                train_writer = csv.writer(train)
                test_writer = csv.writer(test)

                next(reader)

                # Iterate over each row in the input file
                for row in reader:
                    # Randomly assign each row to either the training or testing file
                    if random.random() < 0.8:
                        train_writer.writerow(row)
                    else:
                        test_writer.writerow(row)


indexed_split()
random_split()
