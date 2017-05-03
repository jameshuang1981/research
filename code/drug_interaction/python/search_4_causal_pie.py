

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
# val: "discrete" or "continuous"
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

sample_size_cutoff = 30

spamwriter_log = None

spamwriter_pie = None


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
    load_data(src_file, True, False)

    # Load target file
    load_data(trg_file, False, False)

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
# Flag, indicating whether the var is continuous
# Default is True
cont_F = True
def load_data(data_file, src_F, cont_F):
    with open(data_file, 'r') as f:
        spamreader = list(csv.reader(f, delimiter = ','))

        # Get data_type_Dic, val_Dic, src_Dic and trg_Dic
        # From the second line to the last (since the first line is the header)
        for i in range(1, len(spamreader)):
            for j in range(len(spamreader[0])):
                # var's name lies in jth column in the first row
                var = spamreader[0][j].strip()

                # If the value at [i][j] is not missing
                if spamreader[i][j]:
                    # Get the value
                    val = spamreader[i][j].strip()

                    # Get data_type_Dic
                    if not var in data_type_Dic:
                        if cont_F:
                            data_type_Dic[var] = "continuous"
                        else:
                            data_type_Dic[var] = "discrete"

                    # Get val_Dic
                    if not var in val_Dic:
                        val_Dic[var] = {}
                    if cont_F:
                        val_Dic[var][i] = float(val)
                    else:
                        if val == '1':
                            val_Dic[var][i] = 1
                        else:
                            val_Dic[var][i] = 0

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
        #print(['target:', target])

        # Used for BFS
        q = queue.Queue()

        # Initialization, add all slices to queue
        for i in range(len(slice_LL)):
            q.put([i])

        # BFS
        while not q.empty():
            size = q.qsize()
            for i in range(size):
                # Poll a pie from queue
                pie_L = q.get()

                # Check sufficient condition, i.e., P(target | pie) >> P(target)
                if not check_sff_cnd(target, pie_L, spamwriter_log):
                    # Get the index of the last slice
                    index = pie_L[len(pie_L) - 1]

                    # Visit slices after slice_L[index]
                    for j in range(index + 1, len(slice_LL)):
                        # The new pie including the slice
                        pie_inc_L = pie_L + [j]
                        #if not (['y_T', 1, 1] in decode(pie_inc_L) and ['z_T', 1, 1] in decode(pie_inc_L)):
                            #continue

                        #print(decode(pie_inc_L))
                        # Get sample with respect to [pie_L \wedge [j]] (i.e. the pie including the slice)
                        val_inc_L = get_pie_A_not_B_val_L(target, pie_inc_L, None)
                        #print(['val_inc_L: ', val_inc_L])

                        # Get sample with respect to [pie_L \wedge \neg [j]] (i.e. the pie excluding the slice)
                        val_exc_S_L = get_pie_A_not_B_val_L(target, pie_L, [j])
                        #print(['val_exc_S_L: ', val_exc_S_L])

                        # Get sample with respect to [[j] \wedge \neg pie_L] (i.e. the slice excluding the pie)
                        val_exc_P_L = get_pie_A_not_B_val_L(target, [j], pie_L)
                        #print(['val_exc_P_L: ', val_exc_P_L])

                        #print([len(val_inc_L), len(val_exc_S_L), len(val_exc_P_L)])

                        # Check statistic significance, i.e. # samples >= sample size cutoff
                        if len(val_inc_L) >= sample_size_cutoff and len(val_exc_S_L) >= sample_size_cutoff and len(val_exc_P_L) >= sample_size_cutoff:
                            #print(['after: ', decode(pie_inc_L)])

                            # Get dyn_p_val_cutoff
                            #dyn_p_val_cutoff = get_dynamic_p_val_cutoff(target, pie_inc_L)
                            dyn_p_val_cutoff = 1

                            # Mutual Benefit test
                            spamwriter_log.writerow(['Inclusion'])
                            if MB_test(target, pie_inc_L, [j], val_inc_L, val_exc_S_L, dyn_p_val_cutoff, spamwriter_log) and MB_test(target, pie_inc_L, pie_L, val_inc_L, val_exc_P_L, dyn_p_val_cutoff, spamwriter_log):
                                # Add the new pie to the queue
                                q.put(pie_inc_L)

                # If the pie is sufficient, i.e., P(target | pie) >> P(target)
                else:
                    print(['before: ', decode(pie_L)])
                    spamwriter_log.writerow(['Exclusion'])
                    pie_L = exclusion(target, pie_L, spamwriter_log)
                    if len(pie_L) == 0:
                        continue
                    # Update pie_Dic
                    if target not in pie_Dic:
                        pie_Dic[target] = []
                        pie_Dic[target].append(pie_L)
                    else:
                        # Check whether there is a subset or superset of pie_L in pie_Dic[target]
                        subset_F = False
                        superset_F = False
                        for i in range(len(pie_Dic[target])):
                            piei_L = pie_Dic[target][i]
                            if check_subset(piei_L, pie_L):
                                subset_F = True
                                break
                            elif check_subset(pie_L, piei_L):
                                pie_Dic[target][i] = pie_L
                                superset_F = True
                                break
                        if not subset_F and not superset_F:
                            print(['after: ', decode(pie_L)])
                            # Update pie_Dic
                            pie_Dic[target].append(pie_L)

        # Write pie_file
        if target in pie_Dic:
            for pie_L in pie_Dic[target]:
                spamwriter_pie.writerow([target + ': ', decode(pie_L)])
                # Print the pie
                print([target + ': ', decode(pie_L)])


def get_min_len_int(pie_L):
    min_len_int = None

    if not pie_L or len(pie_L) == 0:
        return min_len_int

    for index in pie_L:
        win_start = slice_LL[index][1]
        win_end = slice_LL[index][2]
        len_intersection = win_end - win_start + 1
        if not min_len_int or min_len_int > len_intersection:
            min_len_int = len_intersection

    return min_len_int


# Check whether there is a set in i_LLL that is a subset of the sets in j_LLL
def check_subset(i_LL, j_LL):
    # Check whether i_LL is a subset of j_LL
    for i_L in i_LL:
        if not i_L in j_LL:
            return False
    return True


def decode(pie_L):
    temp_L = []
    if pie_L:
        for index in pie_L:
            temp_L.append(slice_LL[index])

    return temp_L


def get_val_L_min_len_int(target, min_len_int):
    # Get val_LL
    val_L = []
    temp_L = []

    for time in val_Dic[target]:
        if len(temp_L) < min_len_int:
            temp_L.append(val_Dic[target][time])
        else:
            val_L.append(max(temp_L))
            temp_L = []

    if len(temp_L) > 0:
        val_L.append(max(temp_L))

    return val_L


# Check sufficient condition, i.e., P(target | pie) >> P(target)
def check_sff_cnd(target, pie_L, spamwriter_log):
    # Output log file
    spamwriter_log.writerow(["check_sff_cnd: ", target])
    spamwriter_log.writerow(["target: ", target])
    spamwriter_log.writerow(["pie_L: ", decode(pie_L)])

    # Get val_trg_cnd_pie_L
    val_trg_cnd_pie_L = get_pie_A_not_B_val_L(target, pie_L, None)

    min_len_int = get_min_len_int(pie_L)
    val_L = get_val_L_min_len_int(target, min_len_int)
    #print(val_L)

    #print(['1:', val_trg_cnd_pie_L])
    #print(['2:', val_trg_L])

    # Unpaired t test
    t, p = stats.ttest_ind(val_trg_cnd_pie_L, val_L, equal_var = False)

    # Output log file
    spamwriter_log.writerow(["t: ", t])
    spamwriter_log.writerow(["p: ", p])
    spamwriter_log.writerow(["mean(val_trg_cnd_pie_L): ", np.mean(val_trg_cnd_pie_L)])
    spamwriter_log.writerow(["mean(val_L): ", np.mean(val_L)])
    spamwriter_log.writerow('')

    # If target is discrete
    if data_type_Dic[target] == "discrete":
        # If the pie does not significantly increase the occurrence of the target
        if t <= 0 or p >= p_val_cutoff:
            return False
    # If 1) target is continuous and 2) the pie does not significantly increase or decrease the value of the target
    elif p >= p_val_cutoff:
        return False

    return True

# Get dynamic p_val_cutoff
def get_dynamic_p_val_cutoff(target, pie_L):
    # Get P(target | pie)
    val_L = get_pie_A_not_B_val_L(target, pie_L, None)
    p_target_cnd_pie = np.mean(val_L)

    # Get dynamic p_val_cutoff
    dyn_p_val_cutoff = np.exp(np.log(p_val_cutoff) * p_target_cnd_pie)

    return dyn_p_val_cutoff


# Exclude the pieces that are not in the causal pie
def exclusion(target, pie_L, spamwriter_log):
    # Backup pie_L
    backup_pie_L = [] + pie_L

    # Check each piece
    for index in backup_pie_L:
        # Get pie \ piece
        temp = [] + pie_L
        temp.remove(index)
        if len(temp) == 0:
            return temp

        # Output log file
        spamwriter_log.writerow(["target: ", target])
        spamwriter_log.writerow(["pie_L: ", decode(pie_L)])

        # Get sample with respect to [pie \ piece \wedge \neg piece] (i.e. the pie \ piece excluding the piece)
        val_exc_L = get_pie_A_not_B_val_L(target, temp, [index])

        min_len_int = get_min_len_int(pie_L)
        val_L = get_val_L_min_len_int(target, min_len_int)

        # Unpaired t test
        t, p = stats.ttest_ind(val_exc_L, val_L, equal_var = False)

        # Output log file
        spamwriter_log.writerow(["t: ", t])
        spamwriter_log.writerow(["p: ", p])
        spamwriter_log.writerow(["mean(val_trg_L): ", np.mean(val_L)])
        spamwriter_log.writerow(["mean(val_exc_L): ", np.mean(val_exc_L)])
        spamwriter_log.writerow('')

        # If target is discrete
        if data_type_Dic[target] == "discrete":
            # If the pie does not significantly increase the occurrence of the target
            if t > 0 and p < p_val_cutoff:
                pie_L.remove(index)
        # If 1) target is continuous and 2) the pie does not significantly increase or decrease e's value
        elif p < p_val_cutoff:
            pie_L.remove(index)

    return pie_L


def get_pie_A_not_B_val_L(target, pie_A_L, pie_B_L):
    #print([decode(pie_A_L), decode(pie_B_L)])

    pie_A_time_LL = get_pie_time_LL(pie_A_L)
    pie_B_time_LL = get_pie_time_LL(pie_B_L)
    pie_A_not_B_time_LL = get_pie_A_not_B_time_LL(pie_A_time_LL, pie_B_time_LL)
    #print(['pie_A_time_LL: ', pie_A_time_LL])
    #print(['pie_B_time_LL: ', pie_B_time_LL])
    #print(['pie_A_not_B_time_LL: ', pie_A_not_B_time_LL])

    # Test
    pie_A_print_L = []
    if pie_A_L:
        for index in pie_A_L:
            pie_A_print_L.append(slice_LL[index])
    pie_B_print_L = []
    if pie_B_L:
        for index in pie_B_L:
            pie_B_print_L.append(slice_LL[index])
    # print(["pie_A_L", pie_A_print_L])
    # print(["pie_B_L", pie_B_print_L])
    # print(["pie_A_time_LL", pie_A_time_LL])
    # print(["pie_B_time_LL", pie_B_time_LL])
    # print(["pie_A_not_B_time_LL", pie_A_not_B_time_LL])
    # print

    val_L = get_val_L(target, pie_A_not_B_time_LL)
    return val_L


def get_pie_time_LL(pie_L):
    # Initialization
    pie_time_LL = []

    # If pie_L is None, return pie_time_LL
    if not pie_L:
        return pie_time_LL

    # Get maximum length of intersection
    max_len = None
    for index in pie_L:
        win_start = slice_LL[index][1]
        win_end = slice_LL[index][2]
        length = win_end - win_start + 1
        if not max_len or max_len > length:
            max_len = length

    # Get dictionary of start and end
    [start_Dic, end_Dic] = get_start_end_Dic(pie_L)
    #print([start_Dic, end_Dic])

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
        #print(['len(pie_time_Dic): ', len(pie_time_Dic)])
        if len(pie_time_Dic) == len(pie_L):
            if not recorded_F:
                time_L = []
                recorded_F = True
            time_L.append(time)
            # If the last timepoint or the length of the intersection equals the maximum length
            if time == max_time_stamp or len(time_L) == max_len:
                pie_time_LL.append(time_L)
                recorded_F = False
        # If some slices are absent and we have been recording time
        else:
            if recorded_F:
                pie_time_LL.append(time_L)
                recorded_F = False

    #print(['pie_time_LL:', pie_time_LL])
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
                #start_time = min(time + win_start, max_time_stamp)
                #end_time = min(time + win_end + 1, max_time_stamp)
                start_time = time + win_start
                end_time = time + win_end + 1
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

    # Initialization 
    pie_A_not_B_time_LL = []
    for pie_A_time_L in pie_A_time_LL:
        pie_A_not_B_time_L = []
        for time in pie_A_time_L:
            if not time in pie_B_time_Dic:
                pie_A_not_B_time_L.append(time)
        pie_A_not_B_time_LL.append(pie_A_not_B_time_L)

    return pie_A_not_B_time_LL


# Get the value of target in the time slots
def get_val_L(target, time_LL):
    # Initialization
    val_L = []

    if not time_LL or len(time_LL) == 0:
        return val_L

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
def MB_test(target, pie_inc_L, pie_exc_L, val_inc_L, val_exc_L, dyn_p_val_cutoff, spamwriter_log):
    # Output log file
    spamwriter_log.writerow(["target: ", target])
    spamwriter_log.writerow(["pie_inc_L: ", decode(pie_inc_L)])
    spamwriter_log.writerow(["pie_exc_L: ", decode(pie_exc_L)])
    #spamwriter_log.writerow(["val_inc_L: ", val_inc_L])
    #spamwriter_log.writerow(["val_exc_L: ", val_exc_L])

    # Unpaired t test
    t, p = stats.ttest_ind(val_inc_L, val_exc_L, equal_var = False)

    # Output log file
    spamwriter_log.writerow(["t: ", t])
    spamwriter_log.writerow(["p: ", p])
    spamwriter_log.writerow(["dyn_p_val_cutoff: ", dyn_p_val_cutoff])
    spamwriter_log.writerow(["mean(val_inc_L): ", np.mean(val_inc_L)])
    spamwriter_log.writerow(["mean(val_exc_L): ", np.mean(val_exc_L)])
    spamwriter_log.writerow('')


    # If target is discrete
    if data_type_Dic[target] == "discrete":
        # If compared to exclusion, inclusion does not significantly increase e's occurrence
        #if np.mean(val_inc_L) <= np.mean(val_exc_L) or p >= dyn_p_val_cutoff:
        if p >= dyn_p_val_cutoff:
            # Will not include
            return False
    # If 1) target is continuous and 2) inclusion does not significantly increase or decrease e's value
    elif p >= dyn_p_val_cutoff:
            # Will not include
            return False
    # Include
    return True


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

    with open(log_file, 'w') as f:
        spamwriter_log = csv.writer(f, delimiter = ' ')
        with open(pie_file, 'w') as f:
            spamwriter_pie = csv.writer(f, delimiter = ' ')
            # Get candidates
            search(spamwriter_log, spamwriter_pie)
