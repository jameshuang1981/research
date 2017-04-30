

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


# The dictionary of value
# key: time->var
# val: value of var at the time
val_Dic = {}

# The dictionary of probabiity and causal pie
# key: target
# val: list comprised of probability and the causal pies
prob_causal_pie_L_Dic = {}

# The dictionary of the probability of the variable (source and target) being present
prob_Dic = {}

# Generate source and target data
def generate_data():
    # Load the source setting file
    with open(src_setting_file, 'r') as f:
        spamreader = list(csv.reader(f, delimiter = ','))

        # Get the source list
        src_L = []

        # From the second line to the last (since the first line is the header)
        for i in range(1, len(spamreader)):
            # Get the name of the var and the probability of its presence
            var = spamreader[i][0].strip()
            prob = float(spamreader[i][1].strip())
            prob_Dic[var] = prob
            # Update the source list
            src_L.append(var)

    random.seed()

    # Generate the source value
    for day in range(day_num):
        for source in src_L:
            for hour in range(24):
                # Initialization
                time = day * 24 + hour
                if not time in val_Dic:
                    val_Dic[time] = {}
                val_Dic[time][source] = 0
            # Generate random number from [0, 1)
            rand_prob = random.random()
            if rand_prob < prob_Dic[source]:
                # Generate random number from [0, 23]
                rand_hour = random.randint(0, 23)
                val_Dic[day * 24 + rand_hour][source] = 1

    # Write the source file
    with open(src_file, 'w') as f:
        spamwriter = csv.writer(f, delimiter = ',')
        # Write the header
        spamwriter.writerow(src_L)
        # Write the value
        for time in val_Dic:
            val_L = []
            for source in src_L:
                val_L.append(val_Dic[time][source])
            spamwriter.writerow(val_L)

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

    # Load the target setting file
    with open(trg_setting_file, 'r') as f:
        spamreader = list(csv.reader(f, delimiter = ','))

        # Initialize the target list
        trg_L = []

        # From the second line to the last (since the first line is the header)
        for i in range(1, len(spamreader)):
            # Get the name of the var and the probability of its presence
            var = spamreader[i][0].strip()
            prob = float(spamreader[i][1].strip())
            prob_Dic[var] = prob
            # Update the target list
            trg_L.append(var)

    # Generate the target value
    for target in trg_L:
        # Initialization
        for time in val_Dic:
            rand_prob = random.random()
            if rand_prob < prob_Dic[target]:
                # Add noise
                val_Dic[time][target] = 1
            else:
                val_Dic[time][target] = 0

        # Add the impact of the causal pies
        for time in val_Dic:
            if val_Dic[time][target] == 1:
                continue

            for [prob, causal_pie_LL] in prob_causal_pie_L_Dic[target]:
                if get_presence(time, target, causal_pie_LL):
                    # Generate random number from [0, 1]
                    rand_prob = random.uniform(0, 1)
                    if rand_prob < prob:
                        val_Dic[time][target] = 1
                        break

    # Write the target file
    with open(trg_file, 'w') as f:
        spamwriter = csv.writer(f, delimiter=',')
        # Write the header
        spamwriter.writerow(trg_L)
        # Write the value
        for time in val_Dic:
            val_L = []
            for target in trg_L:
                val_L.append(val_Dic[time][target])
            spamwriter.writerow(val_L)


# Check the presence of each piece in the pie
def get_presence(time, target, causal_pie_LL):
    # Check the presence of each piece in the pie
    for piece_L in causal_pie_LL:
        # Get the name, window start and window end of the piece
        var = piece_L[0]
        win_start = piece_L[1]
        win_end = piece_L[2]

        # Check the presence of the piece in time window [time - window end, time - window start]
        # Default is absence
        presence_F = False
        for prev_time in range(time - win_end, time - win_start + 1):
            if prev_time in val_Dic and val_Dic[prev_time][var] == 1:
                presence_F = True

        # If the piece is absent in the window above
        if not presence_F:
            return False

    # If all the pieces in the pie are present in the corresponding windows
    return True


# Main function
if __name__=="__main__":
    # get parameters from command line
    # please see details of the parameters in the readme file
    src_setting_file = sys.argv[1]
    trg_setting_file = sys.argv[2]
    causal_pie_file = sys.argv[3]
    src_file = sys.argv[4]
    trg_file = sys.argv[5]
    day_num = int(sys.argv[6])

    # Make directory
    directory = os.path.dirname(src_file)
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = os.path.dirname(trg_file)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Generate data
    generate_data()
