

# Please cite the following paper when using the code


# Modules
from __future__ import division
from scipy import stats
import queue
import sys
import os
import csv
import numpy as np
import math


# Notations
# _L      : indicates the data structure is a list
# _LL     : indicates the data structure is a list of list
# _Dic    : indicates the data structure is a dictionary
# _L_Dic  : indicates the data structure is a dictionary, where the value is a list
# _LL_Dic : indicates the data structure is a dictionary, where the value is a list of list


# Define global variables
# The list of time windows, where each window, win, is a list, [win_start, win_end]
win_LL = []

# The dictionary of sources
# key: var
# val: 1
src_Dic = {}

# The dictionary of targets
# key: var
# val: 1
trg_Dic = {}

# The dictionary of time series
# key: time
# val: 1
time_series_Dic = {}

# The list of timepoints
time_series_L = []

# The maximum time stamp
max_time_stamp = 0

# The dictionary of data types
# key: var
# val: "discrete" or "continuous_valued"
data_type_Dic = {}

# The dictionary of value
# key: var->time
# val: value of var at the time
val_Dic = {}

# The list of slices
slice_LL = []

# The dictionary of pies
# key: target
# val: list of pies
pie_Dic = {}


# Initialization
# @param        src_file           source data file, which includes variables that can be the causes, the data are of the following form
#                                  var1 (i.e. header), var1_t1 (i.e. val), ..., var1_tn (i.e. val)
#                                  , ...,
#                                  varn (i.e. header), varn_t1 (i.e. val), ..., varn_tn (i.e. val)
# @param        trg_file           target data file, which includes variables that can be the effects, the data are of the following form
#                                  var1 (i.e. header), var1_t1 (i.e. val), ..., var1_tn (i.e. val)
#                                  , ...,
#                                  varn (i.e. header), varn_t1 (i.e. val), ..., varn_tn (i.e. val)
def initialization(src_file, trg_file):
    # Load source file
    load_data(src_file, True)

    # Load target file
    load_data(trg_file, False)

    # Get windows
    get_win_LL(lag_L)

    # Get slices
    get_slice_LL()

    # Get time series
    get_time_series()

    # Get max time stamp
    global max_time_stamp
    max_time_stamp = time_series_L[len(time_series_L) - 1]


# Load data, get data_type_Dic, val_Dic, src_Dic and trg_Dic
# @param        data_file          source / target file
#                                  the data are of the following form
#                                  time, var1    , ..., varn (i.e. header)
#                                  t1  , var1(t1), ..., varn(t1)
#                                                , ...,
#                                  tn  , var1(tn), ..., varn(tn)
# @param        src_F              Flag variable              
#                                  True,  if target data
#                                  False, if source data
def load_data(data_file, src_F):
    with open(data_file, 'r') as f:
        spamreader = list(csv.reader(f, delimiter = ','))

        # Get data_type_Dic, val_Dic, src_Dic and trg_Dic
        # From the second line to the last (since the first line is the header)
        for i in range(1, len(spamreader)):
            # Time lies in the first column in each row
            time = int(spamreader[i][0].strip())

            # From the second column to the last (since the first column is the time)
            for j in range(1, len(spamreader[0])):
                # Flag, indicating whether the var is continuous_valued
                # Default is True 
                cont_F = True

                # For continuous_valued var, its name is var itself (e.g. Glucose) 
                # var's name lies in jth column in the first row
                var = spamreader[0][j].strip()

                # If the value at [i][j] is not missing
                if spamreader[i][j]:
                    # Get the value
                    val = spamreader[i][j].strip()

                    # If discrete
                    if not is_number(val):
                        # Flip cont_F
                        cont_F = False

                        # For discrete var, its name is var_val (e.g. Insulin_True)
                        var += "_" + val

                    # Get data_type_Dic
                    if not var in data_type_Dic:
                        if cont_F:
                            data_type_Dic[var] = "continuous_valued"
                        else:
                            data_type_Dic[var] = "discrete"

                    # Get val_Dic
                    if not var in val_Dic:
                        val_Dic[var] = {}
                    if cont_F:
                        val_Dic[var][time] = float(val)
                    else:
                        val_Dic[var][time] = 1

                    # If source file
                    if src_F:
                        # Get src_Dic
                        if not var in src_Dic:
                            src_Dic[var] = 1
                    # If target file
                    else:
                        # Get trg_Dic
                        if not var in trg_Dic:
                            trg_Dic[var] = 1


# Check whether string is a number
# @param        val                a string
def is_number(val):
  try:
    float(val)
    return True
  except ValueError:
    return False


# Get windows
def get_win_LL(lag_L):
    for i in range(0, len(lag_L), 2):
        win_L = [int(lag_L[i]), int(lag_L[i + 1])]
        win_LL.append(win_L)


# Get slices
def get_slice_LL():
    for src in src_Dic:
        for win_L in win_LL:
            slice_L = [src, win_L[0], win_L[1]]
            slice_LL.append(slice_L)


def get_time_series():
    # Get time_series_Dic
    for var in val_Dic:
        for time in val_Dic[var]:
            if not time in time_series_Dic:
                time_series_Dic[time] = 1

    # Get time_series_L
    for time in time_series_Dic:
        time_series_L.append(time)

    # Sort time_series_L
    time_series_L.sort()


# Search for the causal pies
def search(spamwriter_log, spamwriter_pie):
    for target in trg_Dic:
        # Used for BFS
        q = queue.Queue()

        # Initialization, add all slices to queue
        for i in range(len(slice_LL)):
            q.put([i])

        # Get val_trg_L
        val_trg_L = []
        for time in val_Dic[target]:
            val_trg_L.append(val_Dic[target][time])

        # BFS
        while not q.empty():
            size = q.qsize()
            for i in range(size):
                # Poll a pie from queue
                pie_L = q.get()
                print(['pie_L:', pie_L])

                # Check sufficient condition, i.e., P(target | pie) >> P(target)
                if not check_sff_cnd(target, pie_L, val_trg_L):
                    # Flag, indicating whether the pie has been expanded
                    # Default is False
                    expand_F = False

                    # Get the index of the last slice
                    index = pie_L[len(pie_L) - 1]

                    # Visit slices after slice_L[index]
                    for j in range(index + 1, len(slice_LL)):
                        # The new pie including the slice
                        pie_inc_L = pie_L + [j]
                        # Get sample with respect to [pie_L \wedge [j]] (i.e. the pie including the slice)
                        val_inc_L = get_pie_A_not_B_val_L(target, pie_inc_L, None)

                        # Get sample with respect to [pie_L \wedge \neg [j]] (i.e. the pie excluding the slice)
                        val_exc_S_L = get_pie_A_not_B_val_L(target, pie_L, [j])

                        # Get sample with respect to [[j] \wedge \neg pie_L] (i.e. the slice excluding the pie)
                        val_exc_P_L = get_pie_A_not_B_val_L(target, [j], pie_L)

                        # Get dyn_p_val_cutoff
                        dyn_p_val_cutoff = get_dynamic_p_val_cutoff(target, pie_inc_L)

                        # Mutual Benefit test
                        if MB_test(target, pie_inc_L, [j], val_inc_L, val_exc_S_L, dyn_p_val_cutoff) and MB_test(target, pie_inc_L, pie_L, val_inc_L, val_exc_P_L, dyn_p_val_cutoff):
                            # Add the new pie to the queue
                            q.put(pie_inc_L)
                            expand_F = True

                # If the pie:
                # either is sufficient, i.e., P(target | pie) >> P(target),
                # or cannot be expanded
                if not expand_F:
                    pie_L = exclusion(target, pie_L, val_trg_L)
                    # If the pie consists of more than one piece
                    if len(pie_L) > 1:
                        # Update pie_Dic
                        if target not in pie_Dic:
                            pie_Dic[target] = []
                        pie_Dic[target].append(pie_L)

                        # Write pie_file
                        pie_print_L = []
                        for index in pie_L:
                            pie_print_L.append(slice_LL[index])
                            spamwriter_pie.writerow([target + ': ', pie_print_L])

                        # Print the pie
                        print([target + ': ', pie_print_L])


# Check sufficient condition, i.e., P(target | pie) >> P(target)
def check_sff_cnd(target, pie_L, val_trg_L):
    # Get val_trg_cnd_pie_L
    val_trg_cnd_pie_L = get_pie_A_not_B_val_L(target, pie_L, None)

    print(['1:', val_trg_cnd_pie_L])
    print(['2:', val_trg_L])


    # Unpaired t test
    t, p = stats.ttest_ind(val_trg_cnd_pie_L, val_trg_L, equal_var = False)


    # If target is discrete
    if data_type_Dic[target] == "discrete":
        # If the pie does not significantly increase the occurrence of the target
        if t <= 0 or p >= p_val_cutoff:
            return False
    # If 1) target is continuous_valued and 2) the pie does not significantly increase or decrease the value of the target
    elif p >= p_val_cutoff:
        return False


# Get dynamic p_val_cutoff
def get_dynamic_p_val_cutoff(target, pie_L):
    # Get P(target | pie)
    val_L = get_pie_A_not_B_val_L(target, pie_L, None)
    p_target_cnd_pie = min(val_L)

    # Get dynamic p_val_cutoff
    dyn_p_val_cutoff = np.exp(np.log(p_val_cutoff) * p_target_cnd_pie)

    return dyn_p_val_cutoff


# Exclude the pieces that are not in the causal pie
def exclusion(target, pie_L, val_trg_L):
    # Backup pie_L
    backup_pie_L = [] + pie_L

    # Check each piece
    for index in backup_pie_L:
        # Get pie \ piece
        temp = [] + pie_L
        temp.remove(index)
        if len(temp) == 0:
            return temp

        # Get sample with respect to [pie \ piece \wedge \neg piece] (i.e. the pie \ piece excluding the piece)
        val_exc_L = get_pie_A_not_B_val_L(target, temp, [index])

        # Unpaired t test
        t, p = stats.ttest_ind(val_exc_L, val_trg_L, equal_var = False)

        # If target is discrete
        if data_type_Dic[target] == "discrete":
            # If the pie does not significantly increase the occurrence of the target
            if t > 0 and p < p_val_cutoff:
                pie_L.remove(index)
        # If 1) target is continuous_valued and 2) the pie does not significantly increase or decrease e's value
        elif p < p_val_cutoff:
            pie_L.remove(index)

    return pie_L


def get_pie_A_not_B_val_L(target, pie_A_L, pie_B_L):
    pie_A_time_LL = get_pie_time_LL(pie_A_L)
    pie_B_time_LL = get_pie_time_LL(pie_B_L)
    pie_A_not_B_time_LL = get_pie_A_not_B_time_LL(pie_A_time_LL, pie_B_time_LL)

    # Test
    pie_A_print_L = []
    if pie_A_L:
        for index in pie_A_L:
            pie_A_print_L.append(slice_LL[index])
    pie_B_print_L = []
    if pie_B_L:
        for index in pie_B_L:
            pie_B_print_L.append(slice_LL[index])
    print(["pie_A_L", pie_A_print_L])
    print(["pie_B_L", pie_B_print_L])
    print(["pie_A_time_LL", pie_A_time_LL])
    print(["pie_B_time_LL", pie_B_time_LL])
    print(["pie_A_not_B_time_LL", pie_A_not_B_time_LL])
    print

    val_L = get_val_L(target, pie_A_not_B_time_LL)
    return val_L


def get_pie_time_LL(pie_L):
    # Initialization
    pie_time_LL = []

    # If pie_L is None, return pie_time_LL
    if not pie_L:
        return pie_time_LL

    # Get dictionary of start and end
    [start_Dic, end_Dic] = get_start_end_Dic(pie_L)
    print([start_Dic, end_Dic])

    # Test
    # print(["start_Dic", start_Dic])
    # print(["end_Dic", end_Dic])

    # Get pie_time_Dic
    # Key: var
    # Value: number of times the var occurs
    pie_time_Dic = {}
    # Flag, indicating whether we have started recording the timepoints where all the slices in the pie are present
    # Default is False
    recorded_F = False

    # Get pie_time_LL 
    for time in time_series_L:
        for index in pie_L:
            if index in start_Dic and time in start_Dic[index]:
                if index in pie_time_Dic:
                    pie_time_Dic[index] += 1
                else:
                    pie_time_Dic[index] = 1
            if index in end_Dic and time in end_Dic[index]:
                pie_time_Dic[index] -= 1
                if pie_time_Dic[index] == 0:
                    del pie_time_Dic[index]
        # If all the slices in the pie are present
        print(['len(pie_time_Dic): ', len(pie_time_Dic)])
        if len(pie_time_Dic) == len(pie_L):
            if not recorded_F:
                time_L = []
                recorded_F = True
            time_L.append(time)
        # If some slices are absent and we have been recording time
        else:
            if recorded_F:
                pie_time_LL.append(time_L)
                recorded_F = False

    # If all the slices in the pie are present and we have been recording time
    if recorded_F:
        pie_time_LL.append(time_L)
        recorded_F = False

    print(['pie_time_LL:', pie_time_LL])
    return pie_time_LL


def get_start_end_Dic(pie_L):
    # Initialization
    start_Dic = {}
    end_Dic = {}

    for index in pie_L:
        [var, win_start, win_end] = slice_LL[index]
        # Initialization
        if not index in start_Dic:
            start_Dic[index] = {}
        if not index in end_Dic:
            end_Dic[index] = {}

        for time in val_Dic[var]:
            if val_Dic[var][time] == 1:
                start_time = min(time + win_start, max_time_stamp)
                end_time = min(time + win_end + 1, max_time_stamp)
                start_Dic[index][start_time] = 1
                end_Dic[index][end_time] = 1
    return [start_Dic, end_Dic]


# Get the time when pie A is present whereas pie B is absent
def get_pie_A_not_B_time_LL(pie_A_time_LL, pie_B_time_LL):
    if not pie_B_time_LL:
        return pie_A_time_LL

    # Get time_Dic
    pie_B_time_Dic = {}
    for pie_B_time_L in pie_B_time_LL:
        for time in pie_B_time_L:
            pie_B_time_Dic[time] = 1

    # Get pie_A_not_B_time_LL
    # Flag, indicating whether we have started recording the timepoints where pie_A is present but pie_B is absent
    recorded_F = False

    # Initialization 
    pie_A_not_B_time_LL = []

    for pie_A_time_L in pie_A_time_LL:
        pie_A_not_B_time_L = []
        for time in pie_A_time_L:
            if not time in pie_B_time_Dic:
                if not recorded_F:
                    pie_A_not_B_time_L = []
                    recorded_F = True
                pie_A_not_B_time_L.append(time)
            else:
                if recorded_F:
                    pie_A_not_B_time_LL.append(pie_A_not_B_time_L)
                    recorded_F = False
        if recorded_F:
            pie_A_not_B_time_LL.append(pie_A_not_B_time_L)
            recorded_F = False
    return pie_A_not_B_time_LL


# Get the value of target in the time slots
def get_val_L(target, time_LL):
    if not time_LL:
        return None

    # Initialization
    val_L = []

    # For each time_L, get the maximum absolute value
    for time_L in time_LL:
        # Initialization
        max_abs_val = None
        for time in time_L:
            if time in val_Dic[target] and (not max_abs_val or abs(max_abs_val) < abs(val_Dic[target][time])):
                max_abs_val = val_Dic[target][time]
        if data_type_Dic[target] == "discrete":
            if not max_abs_val:
                val_L.append(0)
            else:
                val_L.append(max_abs_val)
        else:
            if max_abs_val is not None:
                val_L.append(max_abs_val)
    return val_L


# Mutual Benefit test
def MB_test(target, pie_inc_L, pie_exc_L, val_inc_L, val_exc_L, dyn_p_val_cutoff):
    # Output log file
    with open(log_file, 'a') as f:
        spamwriter = csv.writer(f, delimiter = ' ')

        spamwriter.writerow(["target: ", target])

        pie_inc_print_L = []
        for index in pie_inc_L:
            pie_inc_print_L.append(slice_LL[index])
        spamwriter.writerow(["pie_inc_L: ", pie_inc_print_L])

        pie_exc_print_L = []
        for index in pie_exc_L:
            pie_exc_print_L.append(slice_LL[index])
        spamwriter.writerow(["pie_exc_L: ", pie_exc_print_L])

        spamwriter.writerow(["val_inc_L: ", val_inc_L])

        spamwriter.writerow(["val_exc_L: ", val_exc_L])

    if not val_inc_L or not val_exc_L:
        return False

    # Unpaired t test
    t, p = stats.ttest_ind(val_inc_L, val_exc_L, equal_var = False)

    # Output log file
    with open(log_file, 'a') as f:
        spamwriter = csv.writer(f, delimiter = ' ')

        spamwriter.writerow(["t: ", t])

        spamwriter.writerow(["p: ", p])

    # If target is discrete
    if data_type_Dic[target] == "discrete":
        # If compared to exclusion, inclusion does not significantly increase e's occurrence
        if t <= 0 or p >= p_val_cutoff:
            # Will not include
            return False
    # If 1) target is continuous_valued and 2) inclusion does not significantly increase or decrease e's value
    elif p >= p_val_cutoff:
            # Will not include
            return False
    # Include
    return True


# Output candidates
def output_candidates():
    with open(pie_file, 'wb') as f:
        spamwriter = csv.writer(f, delimiter = ',')
        for trg in pie_Dic:
            if pie_Dic[trg]:
                for pie in pie_Dic[trg]:
                    pie_print_L = []
                    for index in pie:
                        pie_print_L.append(slice_LL[index])
                    spamwriter.writerow([trg, pie_print_L])


# Main function
if __name__=="__main__":
    # get parameters from command line
    # please see details of the parameters in the readme file
    src_file = sys.argv[1]
    trg_file = sys.argv[2]
    pie_file = sys.argv[3]
    log_file = sys.argv[4]
    pie_size_cutoff = int(sys.argv[5])
    p_val_cutoff = float(sys.argv[6])
    sample_size_cutoff = int(sys.argv[7])
    lag_L = sys.argv[8:]

    # Make directory
    directory = os.path.dirname(pie_file)
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = os.path.dirname(log_file)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Initialization
    initialization(src_file, trg_file)

    # Write log file
    with open(log_file, 'w') as f:
        spamwriter_log = csv.writer(f, delimiter = ' ')
        spamwriter_log.writerow("This is the log file.")

        # Write pie file
        with open(pie_file, 'w') as f:
            spamwriter_pie = csv.writer(f, delimiter = ' ')
            spamwriter_pie.writerow("This is the pie file.")

            # Get candidates
            search(spamwriter_log, spamwriter_pie)
