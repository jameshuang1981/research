

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


# Generate the script
def generate_script():
    # Write the script file
    with open(scr_file, 'w') as f:
        script = 'python3 search_4_causal_pie.py ' + src_data_dir + src_data_file + ' ' + tar_data_file + ' ' + pie_file + ' ' + log_file + ' ' + fig_dir_num + ' ' + pie_size_cutoff + ' ' + p_val_cutoff + ' ' + sample_size_cutoff
        for lag in lag_L:
            script += ' ' + lag
        # Write the file
        f.write(script + '\n')

# Main function
if __name__=="__main__":
    # Get parameters from command line
    # Please see details of the parameters in the readme file
    scr_dir = sys.argv[1]
    src_data_dir = sys.argv[2]
    tar_data_dir = sys.argv[3]
    pie_dir = sys.argv[4]
    log_dir = sys.argv[5]
    fig_dir = sys.argv[6]
    pie_size_cutoff = sys.argv[7]
    p_val_cutoff = sys.argv[8]
    sample_size_cutoff = sys.argv[9]
    lag_L = sys.argv[10:]

    # Make directory
    directory = os.path.dirname(scr_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = os.path.dirname(pie_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = os.path.dirname(log_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)


    for src_data_file in os.listdir(src_data_dir):
        if src_data_file.endswith(".txt"):
            # Get src setting file number
            num = src_data_file
            num = num.replace('src_data_', '')
            num = num.replace('.txt', '')
            # Get script file
            scr_file = scr_dir + 'script_' + num + '.txt'
            # Get tar data file
            tar_data_file = tar_data_dir + 'tar_data_' + num + '.txt'
            # Get pie file
            pie_file = pie_dir + 'pie_' + num + '.txt'
            # Get log file
            log_file = log_dir + 'log_' + num + '.txt'
            # Get fig_dir
            fig_dir_num = fig_dir + num + '/'
            directory = os.path.dirname(fig_dir_num)
            if not os.path.exists(directory):
                os.makedirs(directory)

            # Generate the script
            generate_script()

