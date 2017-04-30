

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


# Generate the setting
def generate_setting(type):
    # Write the setting file
    with open(setting_file, 'w') as f:
        spamwriter = csv.writer(f, delimiter=',')
        # Write the header
        spamwriter.writerow(['var', 'probability'])

        random.seed()

        # Get var number
        var_num = random.randint(var_num_range_L[0], var_num_range_L[1])

        for i in range(var_num):
            # Get the probability
            prob = random.uniform(prob_range_L[0], prob_range_L[1])

            # Write the var and probability
            spamwriter.writerow([type + '_' + str(i), prob])


# Main function
if __name__=="__main__":
    # Get parameters from command line
    # Please see details of the parameters in the readme file
    setting_file = sys.argv[1]
    var_num_range_L = [int(sys.argv[2]), int(sys.argv[3])]
    prob_range_L = [float(sys.argv[4]), float(sys.argv[5])]
    type = sys.argv[6]

    # Make directory
    directory = os.path.dirname(setting_file)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Generate the setting
    generate_setting(type)
