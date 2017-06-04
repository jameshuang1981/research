

# Please cite the following paper when using the code


# Modules
import sys
import os
import csv
import numpy as np
import math
import random


# Notations
# _L      : indicates the data structure is a list
# _LL     : indicates the data structure is a list of list
# _Dic    : indicates the data structure is a dictionary
# _L_Dic  : indicates the data structure is a dictionary, where the value is a list
# _LL_Dic : indicates the data structure is a dictionary, where the value is a list of list


# Generate statistics
def generate_statistics():
    # Initialize prob_causal_pie_L_Dic and pie_Dic
    prob_causal_pie_L_Dic = {}
    pie_Dic = {}

    # Load the causal pie file
    with open(causal_pie_file, 'r') as f:
        spamreader = list(csv.reader(f, delimiter = ','))
        # Get the target, probability and causal pie
        # From the second line to the last (since the first line is the header)
        for i in range(1, len(spamreader)):
            # Target lies in the first column in each row
            target = spamreader[i][0].strip()
            # Probability lies in the second column in each row
            prob = float(spamreader[i][1].strip())
            # Causal pie lies in the remaining columns, with the form piece_i, win_start_i, win_end_i
            causal_pie_LL = []
            piece_num = (len(spamreader[i]) - 2) // 3
            for j in range(piece_num):
                piece_L = []
                # Name
                piece_L.append(spamreader[i][j * 3 + 2].strip())
                # Window start
                piece_L.append(int(spamreader[i][j * 3 + 3].strip()))
                # Window end
                piece_L.append(int(spamreader[i][j * 3 + 4].strip()))
                causal_pie_LL.append(piece_L)
            if not target in prob_causal_pie_L_Dic:
                prob_causal_pie_L_Dic[target] = []
            prob_causal_pie_L_Dic[target].append([prob, causal_pie_LL])

    # Load the pie file
    with open(pie_file, 'r') as f:
        spamreader = list(csv.reader(f, delimiter = ' '))
        # Get the target and causal pie
        for i in range(len(spamreader)):
            # Target lies in the end of the first column in each row
            target = spamreader[i][0].strip()
            target = target.replace('causal pie of ', '')
            target = target.replace(':', '')
            # Pie lies in the second column in each row
            pie = spamreader[i][1].strip()
            pie = pie.replace('[', '')
            pie = pie.replace(']', '')
            pie = pie.replace('\'', '')
            pie = pie.split(',')
            piece_num = len(pie) // 3
            pie_LL = []
            for j in range(piece_num):
                piece_L = []
                # Name
                piece_L.append(pie[j * 3].strip())
                # Window start
                piece_L.append(pie[j * 3 + 1].strip())
                # Window end
                piece_L.append(pie[j * 3 + 2].strip())
                pie_LL.append(piece_L)
            if not target in pie_Dic:
                pie_Dic[target] = []
            pie_Dic[target].append(pie_LL)

    # Get true positive and false positive for the current dataset
    tp = 0
    fp = 0
    # For each target
    for target in pie_Dic:
        # For each pie
        for pie_LL in pie_Dic[target]:
            # Flag, indicating whether the pie is a causal pie
            equal_F = False
            if target in prob_causal_pie_L_Dic:
                # For each causal pie and the probability
                for prob, causal_pie_LL in prob_causal_pie_L_Dic[target]:
                    # If the pie is a causal pie
                    if equal(causal_pie_LL, pie_LL):
                        equal_F = True
                        break
            # If the pie is a causal pie
            if equal_F is True:
                # Increase true positive
                tp += 1
            else:
                # Increase false positive
                fp += 1

    # Get false negative
    fn = 0
    # For each target
    for target in prob_causal_pie_L_Dic:
        # For each pie
        for prob, causal_pie_LL in prob_causal_pie_L_Dic[target]:
            # Flag, indicating whether the causal pie has been discovered
            equal_F = False
            if target in pie_Dic:
                # For each pie
                for pie_LL in pie_Dic[target]:
                    # If the causal pie has been discovered
                    if equal(pie_LL, causal_pie_LL):
                        equal_F = True
                        break
            # If the causal pie has not been discovered
            if equal_F is False:
                # Increase false negative
                fn += 1

    return [tp, fp, fn]


# Check whether the two pies are equal
def equal(pie_i_LL, pie_j_LL):
    # The two pies are equal if one belongs to another, and vice versa
    if belong(pie_i_LL, pie_j_LL) is True and belong(pie_j_LL, pie_i_LL) is True:
        return True
    else:
        return False


# Check whether pie_i belongs to pie_j
def belong(pie_i_LL, pie_j_LL):
    # If pie_i is None or empty
    if pie_i_LL is None or len(pie_i_LL) == 0:
        return True
    # If pie_j is None or empty
    elif pie_j_LL is None or len(pie_j_LL) == 0:
        return False

    # For each variable in pie_i
    for var_i, win_start_i, win_end_i in pie_i_LL:
        # Flag, indicating whether var_i is in pie_j
        belong_F = False

        # For each variable in pie_j
        for var_j, win_start_j, win_end_j in pie_j_LL:
            if var_i == var_j:
                belong_F = True
                break

        # If var_i is not in pie_j
        if belong_F is False:
            return False

    return True


# Main function
if __name__=="__main__":
    # get parameters from command line
    # please see details of the parameters in the readme file
    causal_pie_dir = sys.argv[1]
    pie_dir = sys.argv[2]
    statistics_file = sys.argv[3]

    # Make directory
    directory = os.path.dirname(statistics_file)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Initialize true positve, false positive, and false negative (across all datasets)
    tp_all = 0
    fp_all = 0
    fn_all = 0

    # Write statistics file
    with open(statistics_file, 'w') as f:
        for causal_pie_file in os.listdir(causal_pie_dir):
            if causal_pie_file.endswith(".txt"):
                # Get src setting file number
                num = causal_pie_file
                num = num.replace('causal_pie_', '')
                num = num.replace('.txt', '')
                # Get causal_pie_file
                causal_pie_file = causal_pie_dir + causal_pie_file
                # Get pie file
                pie_file = pie_dir + 'pie_' + num + '.txt'

                # Generate statistics
                [tp, fp, fn] = generate_statistics()

                # Write statistics file
                # Write the name of the dataset
                f.write('dataset_' + num + '\n')
                # Write true positive, false positive and false negative for the current dataset
                f.write('tp: ' + str(tp) + '\n')
                f.write('fp: ' + str(fp) + '\n')
                f.write('fn: ' + str(fn) + '\n')
                f.write('\n\n')

                # Update true positive, false positive and false negative across all datasets
                tp_all += tp
                fp_all += fp
                fn_all += fn

        # Write statistics file
        # Write true positive, false positive and false negative across all datasets
        f.write('tp_all: ' + str(tp_all) + '\n')
        f.write('fp_all: ' + str(fp_all) + '\n')
        f.write('fn_all: ' + str(fn_all) + '\n')
        # Write precision and recall across all datasets
        f.write('precision: ' + str(tp_all / (tp_all + fp_all)) + '\n')
        f.write('recall: ' + str(tp_all / (tp_all + fn_all)) + '\n')