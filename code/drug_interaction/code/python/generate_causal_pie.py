

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


# Generate causal pie
def generate_causal_pie():
    # Read source setting file
    with open(src_setting_file, 'r') as f:
        spamreader = list(csv.reader(f, delimiter = ','))
        # Get src_L
        src_L = []
        # From the second line to the last (since the first line is the header)
        for i in range(1, len(spamreader)):
            src_L.append(spamreader[i][0].strip())

    # Read target setting file
    with open(tar_setting_file, 'r') as f:
        spamreader = list(csv.reader(f, delimiter = ','))
        # Get tar_L
        tar_L = []
        # From the second line to the last (since the first line is the header)
        for i in range(1, len(spamreader)):
            tar_L.append(spamreader[i][0].strip())

    # Write causal pie file
    with open(causal_pie_file, 'w') as f:
        spamwriter = csv.writer(f, delimiter = ',')
        # Write the header
        spamwriter.writerow(['target', 'probability', 'causal pie'])

        for target in tar_L:
            # The causal pies
            causal_pie_LLL = []

            # Get the causal pie number
            causal_pie_num = random.randint(causal_pie_num_range_L[0], causal_pie_num_range_L[1])

            while len(causal_pie_LLL) < causal_pie_num:
                # Get the probability
                prob = random.uniform(prob_range_L[0], prob_range_L[1])

                # Get the causal pie
                causal_pie_LL = []

                # Get the number of pieces
                piece_num = random.randint(piece_num_range_L[0], piece_num_range_L[1])

                # Get the name of the pieces
                var_L = []
                while len(var_L) < piece_num:
                    rand_idx = random.randint(0, len(src_L) - 1)
                    if not src_L[rand_idx] in var_L:
                        var_L.append(src_L[rand_idx])

                # Get the window of the pieces, where win_end > win_start
                win_LL = []
                for j in range(piece_num):
                    win_start = random.randint(win_range_L[0], win_range_L[1] // 2)
                    win_end = random.randint(win_start + 1, win_range_L[1])
                    win_LL.append([win_start, win_end])

                # Add the pieces to the causal pie
                for j in range(piece_num):
                    var = var_L[j]
                    win_start = win_LL[j][0]
                    win_end = win_LL[j][1]
                    causal_pie_LL.append([var, win_start, win_end])

                # Check whether the causal pie intersects with the existing ones
                if not check_intersect(causal_pie_LL, causal_pie_LLL):
                    causal_pie_LLL.append(causal_pie_LL)
                    # Write the target, probability, and the causal pie
                    causal_pie_L = []
                    for [var, win_start, win_end] in causal_pie_LL:
                        causal_pie_L.append(var)
                        causal_pie_L.append(win_start)
                        causal_pie_L.append(win_end)
                    spamwriter.writerow([target, prob] + causal_pie_L)


# Check whether i_LL and j_LLL intersect, that is, whether there are two pies containing sources with the same name
def check_intersect(i_LL, j_LLL):
    for i_L in i_LL:
        for j_LL in j_LLL:
            for j_L in j_LL:
                # Check whether i_L[0] equals j_L[0], i.e., the name of the source is the same
                if j_L[0] == i_L[0]:
                    return True

    return False


# Main function
if __name__=="__main__":
    # get parameters from command line
    # please see details of the parameters in the readme file
    src_setting_dir = sys.argv[1]
    tar_setting_dir = sys.argv[2]
    causal_pie_dir = sys.argv[3]
    causal_pie_num_range_L = [int(sys.argv[4]), int(sys.argv[5])]
    piece_num_range_L = [int(sys.argv[6]), int(sys.argv[7])]
    win_range_L = [int(sys.argv[8]), int(sys.argv[9])]
    prob_range_L = [float(sys.argv[10]), float(sys.argv[11])]

    # Make directory
    directory = os.path.dirname(causal_pie_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)

    for src_setting_file in os.listdir(src_setting_dir):
        if src_setting_file.endswith(".txt"):
            # Get src setting file number
            num = src_setting_file
            num = num.replace('src_setting_', '')
            num = num.replace('.txt', '')
            # Get src setting file
            src_setting_file = src_setting_dir + 'src_setting_' + num + '.txt'
            # Get tar setting file
            tar_setting_file = tar_setting_dir + 'tar_setting_' + num + '.txt'
            # Get causal pie file
            causal_pie_file = causal_pie_dir + 'causal_pie_' + num + '.txt'
            # Generate causal pie
            generate_causal_pie()


