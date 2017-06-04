

# Please cite the following paper when using the code


# Modules
from __future__ import division
from scipy import stats
import sys
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import random


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

# The dictionary of value
# key: var->time
# val: value of var at the time
val_Dic = {}

# The dictionary of var
# key: time
# val: the vars occur at the time
var_Dic = {}

# The list of slices
slice_LL = []

# The dictionary of pies
# key: target
# val: list of pies
pie_Dic = {}

sample_size_cutoff = 30

fig_num = 0


# Initialization
# @param        src_data_file           source data file, which includes variables that can be the causes, the data are of the following form
#                                  var1 (i.e. header), var1_t1 (i.e. val), ..., var1_tn (i.e. val)
#                                  , ...,
#                                  varn (i.e. header), varn_t1 (i.e. val), ..., varn_tn (i.e. val)
# @param        trg_data_file           target data file, which includes variables that can be the effects, the data are of the following form
#                                  var1 (i.e. header), var1_t1 (i.e. val), ..., var1_tn (i.e. val)
#                                  , ...,
#                                  varn (i.e. header), varn_t1 (i.e. val), ..., varn_tn (i.e. val)
def initialization(src_data_file, trg_data_file):
    # Load source file
    load_data(src_data_file, True)

    # Load target file
    load_data(trg_data_file, False)

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
            if not i in var_Dic:
                var_Dic[i] = []
            for j in range(len(spamreader[0])):
                # var's name lies in jth column in the first row
                var = spamreader[0][j].strip()

                # If the value at [i][j] is not missing
                if spamreader[i][j]:
                    # Get the value
                    val = spamreader[i][j].strip()

                    # Get val_Dic
                    if not var in val_Dic:
                        val_Dic[var] = {}
                    if val == '1':
                        val_Dic[var][i] = 1
                        var_Dic[i].append(var)
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


# Get the time series
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
def search():
    for target in trg_Dic:
        # Write target to spamwriter_log
        spamwriter_log.writerow(['search ' + target + ': ', target])

        # The list of slices in the pie
        pie_L = []

        # The list of list of timepoints where the target can be changed by the pie
        # Initialization
        tar_con_pie_time_LL = get_tar_con_pie_time_LL(target, pie_L)

        # The first slice added to the pie
        root = None

        # The dictionary records the visited slice
        # key: slice
        # val: 1
        visited_Dic = {}

        # The dictionary records the removed slice relative to the root
        # key: root->slice
        # val: 1
        removed_Dic = {}

        # The loop continues if there are unvisited nodes
        while len(visited_Dic) < len(slice_LL):
            # The loop continues if the pie is not sufficient (to produce the effect)
            while check_suf_con(target, pie_L, tar_con_pie_time_LL) is False:
                [pie_L, tar_con_pie_time_LL, root, min_slice] = expand(target, pie_L, tar_con_pie_time_LL, root, visited_Dic, removed_Dic)
                # If the pie cannot be expanded anymore
                if min_slice is None:
                    break

            # If the pie can be expanded, (i.e. it is sufficient)
            if min_slice is not None:
                # Mark each slice in the pie as unvisited (i.e., deleting the key from the dict), except for the root
                for index in pie_L:
                    if index != root:
                        del visited_Dic[index]

                # Check the necessary condition (to produce the effect) and remove unnecessary slices
                pie_L = check_nec_con(target, pie_L)

                # Mark each slice in the pie as visited (i.e, adding the key to the dict)
                for index in pie_L:
                    visited_Dic[index] = 1

                # Remove the influence of the pie from the data
                remove_inf(target, tar_con_pie_time_LL)

                # Write the pie to spamwriter_pie
                spamwriter_pie.writerow(['causal pie of ' + target + ': ', decode(pie_L)])
                # Output the pie
                print(['causal pie of ' + target + ': ', decode(pie_L)])

                # Clear
                pie_L = []
                tar_con_pie_time_LL = get_tar_con_pie_time_LL(target, pie_L)
                root = None
            else:
                # Shrink
                [pie_L, tar_con_pie_time_LL, max_slice] = shrink(target, pie_L, root, visited_Dic, removed_Dic)

                # If the pie cannot be shrinked anymore
                if max_slice is None:
                    # Mark each slice in the pie as unvisited (i.e., deleting the key from the dict), except for the root
                    for index in pie_L:
                        if index != root:
                            del visited_Dic[index]

                    # Clear
                    pie_L = []
                    tar_con_pie_time_LL = get_tar_con_pie_time_LL(target, pie_L)
                    root = None


# Get the list of list of timepoints where the target can be changed by the pie
def get_tar_con_pie_time_LL(target, pie_L):
    # Initialization
    tar_con_pie_time_LL = []

    # If the pie is None or empty, return the list of list of timepoints where the target is measured
    if pie_L is None or len(pie_L) == 0:
        for time in val_Dic[target]:
            tar_con_pie_time_LL.append([time])
        return tar_con_pie_time_LL

    # Get the minimum window length of slices in the pie
    min_win_len = get_min_win_len(pie_L)

    # Get the dictionary of start and end
    [start_Dic, end_Dic] = get_start_end_Dic(pie_L)

    # Get pie_time_Dic
    # key: var
    # val: number of times the var occurs
    pie_time_Dic = {}

    # Flag, indicating whether we have started recording the timepoints where all the slices in the pie are present
    # Default is False
    recorded_F = False

    # Get tar_con_pie_time_LL
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
        if len(pie_time_Dic) == len(pie_L):
            if recorded_F is False:
                time_L = []
                recorded_F = True
            time_L.append(time)
            # If the last timepoint or the length of the intersection equals the minimum window length
            if time == max_time_stamp or len(time_L) == min_win_len:
                tar_con_pie_time_LL.append(time_L)
                recorded_F = False
        # If some slices are absent and we have been recording time
        elif recorded_F:
            tar_con_pie_time_LL.append(time_L)
            recorded_F = False

    return tar_con_pie_time_LL


# Get the dictionary of window start and window end
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
            # If var occurs at this time
            if val_Dic[var][time] == 1:
                start_time = time + win_start
                end_time = time + win_end + 1
                # Update the two dictionaries
                start_Dic[index][start_time] = 1
                end_Dic[index][end_time] = 1

    return [start_Dic, end_Dic]


# Check sufficient condition, i.e., P(target | pie) >> P(target)
def check_suf_con(target, pie_L, tar_con_pie_time_LL):
    # Output log file
    spamwriter_log.writerow(["check_suf_con target: ", target])
    spamwriter_log.writerow(["check_suf_con pie_L: ", decode(pie_L)])

    # If the pie is None or empty
    if pie_L is None or len(pie_L) == 0:
        return False

    # Get the list of target's value that can be changed by the pie
    tar_con_pie_val_L = get_tar_val_L_time_slot(target, tar_con_pie_time_LL)

    # If not enough sample
    if len(tar_con_pie_val_L) <= sample_size_cutoff:
        return False

    # Get average length of time slot
    ave_len_time_slot = get_ave_len_time_slot(target, tar_con_pie_time_LL)

    # Get the list of target's value relative to the average length of time slot
    tar_val_L = get_tar_val_L_ave_len_time_slot(target, ave_len_time_slot)

    # Unpaired t test
    t, p = stats.ttest_ind(tar_con_pie_val_L, tar_val_L, equal_var = False)

    # Output log file
    spamwriter_log.writerow(["check_suf_con t: ", t])
    spamwriter_log.writerow(["check_suf_con p: ", p])
    spamwriter_log.writerow(["check_suf_con mean(tar_con_pie_val_L): ", np.mean(tar_con_pie_val_L)])
    spamwriter_log.writerow(["check_suf_con mean(tar_val_L): ", np.mean(tar_val_L)])
    spamwriter_log.writerow('')

    # If the pie does not significantly increase the occurrence of the target
    if t <= 0 or p >= p_val_cutoff:
        return False

    # Check each slice that does not belong to the pie
    for index in range(len(slice_LL)):
        if index in pie_L:
            continue

        # Output log file
        spamwriter_log.writerow(["check_suf_con slice_LL[index]: ", slice_LL[index]])

        # Get the list of target's value that can be changed by the slice
        tar_con_sli_time_LL = get_tar_con_pie_time_LL(target, [index])

        # Get the list of list of timepoints where the target can be changed by the pie but not the slice
        tar_con_pie_not_sli_time_LL = get_tar_con_pie_not_sli_time_LL(tar_con_pie_time_LL, tar_con_sli_time_LL)

        # Get the list of target's value that can be changed by the pie but not the slice
        tar_con_pie_not_sli_val_L = get_tar_val_L_time_slot(target, tar_con_pie_not_sli_time_LL)

        # If not enough sample
        if len(tar_con_pie_not_sli_val_L) <= sample_size_cutoff:
            continue

        # Get average length of time slot
        ave_len_time_slot = get_ave_len_time_slot(target, tar_con_pie_not_sli_time_LL)

        # Get the list of target's value relative to the average length of time slot
        tar_val_L = get_tar_val_L_ave_len_time_slot(target, ave_len_time_slot)

        # Unpaired t test
        t, p = stats.ttest_ind(tar_con_pie_not_sli_val_L, tar_val_L, equal_var = False)

        # Output log file
        spamwriter_log.writerow(["check_suf_con t: ", t])
        spamwriter_log.writerow(["check_suf_con p: ", p])
        spamwriter_log.writerow(["check_suf_con mean(tar_con_pie_not_sli_val_L): ", np.mean(tar_con_pie_not_sli_val_L)])
        spamwriter_log.writerow(["check_suf_con mean(tar_val_L): ", np.mean(tar_val_L)])
        spamwriter_log.writerow('')

        # If pie \ slice does not significantly increase the occurrence of the target
        if t <= 0 or p >= p_val_cutoff:
            # Get the list of timepoints where the target can be changed by the pie and the slice
            tar_con_pie_sli_time_LL = get_tar_con_pie_sli_time_LL(target, pie_L, tar_con_pie_time_LL, index)

            # Get the list of target's value that can be changed by the pie and the slice
            tar_con_pie_sli_val_L = get_tar_val_L_time_slot(target, tar_con_pie_sli_time_LL)

            # If not enough sample
            if len(tar_con_pie_sli_val_L) <= sample_size_cutoff:
                continue

            # Unpaired t test
            t, p = stats.ttest_ind(tar_con_pie_sli_val_L, tar_con_pie_not_sli_val_L, equal_var=False)

            # Output log file
            spamwriter_log.writerow(["check_suf_con t: ", t])
            spamwriter_log.writerow(["check_suf_con p: ", p])
            spamwriter_log.writerow(["check_suf_con mean(tar_con_pie_sli_val_L): ", np.mean(tar_con_pie_sli_val_L)])
            spamwriter_log.writerow(["check_suf_con mean(tar_con_pie_not_sli_val_L): ", np.mean(tar_con_pie_not_sli_val_L)])
            spamwriter_log.writerow('')

            # If pie \wedge slice significantly increases the occurrence of the target
            if t > 0 and p < p_val_cutoff:
                return False

    return True


# Get the actual slices in the pie
def decode(pie_L):
    temp_L = []

    # If the pie is None or empty
    if pie_L is None or len(pie_L) == 0:
        return temp_L

    for index in pie_L:
        temp_L.append(slice_LL[index])

    return temp_L


# Get the value of target in the time slots
def get_tar_val_L_time_slot(target, time_LL):
    # Initialization
    tar_val_L = []

    if time_LL is None or len(time_LL) == 0:
        return tar_val_L

    # For each time_L, get the maximum absolute value
    for time_L in time_LL:
        # Get temp_L
        # Initialization
        temp_L = []
        for time in time_L:
            if time in val_Dic[target]:
                temp_L.append(val_Dic[target][time])

        if len(temp_L) == 0:
            continue

        # If temp_L does not contain removed value of the target
        if min(temp_L) != -1:
            # Add the maximum value in the list (so that if the target occurs in the window, it counts as occurred in the window)
            tar_val_L.append(max(temp_L))

    return tar_val_L


# Get the average length of time slot
def get_ave_len_time_slot(target, time_LL):
    # Initialization
    len_L = []

    # If time_LL is None or empty
    if time_LL is None or len(time_LL) == 0:
        return 1

    # For each time_L, get the maximum absolute value
    for time_L in time_LL:
        # Get temp_L
        # Initialization
        temp_L = []
        for time in time_L:
            if time in val_Dic[target]:
                temp_L.append(val_Dic[target][time])

        if len(temp_L) == 0:
            continue

        # If temp_L does not contain removed value of the target
        if min(temp_L) != -1:
            # Add the length of the list
            len_L.append(len(temp_L))

    return np.mean(len_L)


# Get the list of target's value relative to the average length of time slot
def get_tar_val_L_ave_len_time_slot(target, ave_len_time_slot):
    # Initialization
    # tar_val_L is the return value (i.e. the list of value)
    tar_val_L = []
    # temp_L is the list of value in windows no wider than the average length
    temp_L = []

    # For each timepoint where the target is measured
    for time in val_Dic[target]:
        if len(temp_L) == 0:
            # Get len_time_slot
            len_time_slot = int(ave_len_time_slot)
            dif = ave_len_time_slot - len_time_slot
            ran = random.uniform(0, 1)
            if ran <= dif:
                len_time_slot += 1

        # If the length of the list is still narrower than the length of time slot, add the value
        if len(temp_L) < len_time_slot:
            temp_L.append(val_Dic[target][time])
        # If the length of the list is not narrower than the length of time slot
        else:
            # If temp_L does not contain removed value of the target
            if min(temp_L) != -1:
                # Add the maximum value in the list (so that if the target occurs in the window, it counts as occurred in the window)
                tar_val_L.append(max(temp_L))
            # Reset the list
            temp_L = []

    # If the list is not empty
    if len(temp_L) > 0:
        # If temp_L does not contain removed value of the target
        if min(temp_L) != -1:
            # Add the maximum value in the list (so that if the target occurs in the window, it counts as occurred in the window)
            tar_val_L.append(max(temp_L))

    return tar_val_L


# Get the minimum window length of slices in the pie
def get_min_win_len(pie_L):
    # Initialization
    min_win_len = None

    # If the pie is None or empty, return the minimum window length, 1
    if pie_L is None or len(pie_L) == 0:
        return 1

    # For each slice in the pie
    for index in pie_L:
        # Get window start, window end, and the length
        win_start = slice_LL[index][1]
        win_end = slice_LL[index][2]
        win_len = win_end - win_start + 1
        # Update the minimum length
        if min_win_len is None or min_win_len > win_len:
            min_win_len = win_len

    return min_win_len


# Get the list of target's value that can be changed by the pie but not the slice
def get_tar_con_pie_not_sli_val_L(target, tar_con_pie_time_LL, index):
    # Get the list of target's value that can be changed by the slice
    tar_con_sli_time_LL = get_tar_con_pie_time_LL(target, [index])

    # Get the list of list of timepoints where the target can be changed by the pie but not the slice
    tar_con_pie_not_sli_time_LL = get_tar_con_pie_not_sli_time_LL(tar_con_pie_time_LL, tar_con_sli_time_LL)

    # Get the list of target's value that can be changed by the pie but not the slice
    tar_con_pie_not_sli_val_L = get_tar_val_L_time_slot(target, tar_con_pie_not_sli_time_LL)

    return tar_con_pie_not_sli_val_L


# Get the list of list of timepoints where the target can be changed by the pie but not the slice
def get_tar_con_pie_not_sli_time_LL(tar_con_pie_time_LL, tar_con_sli_time_LL):
    if tar_con_sli_time_LL is None or len(tar_con_sli_time_LL) == 0:
        return tar_con_pie_time_LL

    # Get tar_con_sli_time_Dic
    tar_con_sli_time_Dic = {}
    for tar_con_sli_time_L in tar_con_sli_time_LL:
        for time in tar_con_sli_time_L:
            tar_con_sli_time_Dic[time] = 1

    # Initialization
    tar_con_pie_not_sli_time_LL = []
    for tar_con_pie_time_L in tar_con_pie_time_LL:
        tar_con_pie_not_sli_time_L = []
        for time in tar_con_pie_time_L:
            if not time in tar_con_sli_time_Dic:
                tar_con_pie_not_sli_time_L.append(time)
        if len(tar_con_pie_not_sli_time_L) > 0:
            tar_con_pie_not_sli_time_LL.append(tar_con_pie_not_sli_time_L)

    return tar_con_pie_not_sli_time_LL


# Expand the pie by adding the slice that yields the minimum probalbity of the target that can be changed by the pie but not the slice
def expand(target, pie_L, tar_con_pie_time_LL, root, visited_Dic, removed_Dic):
    # This is the slice that yields the minimum probability of the target that can be changed by the pie but not the slice
    min_slice = None
    # This is the probability of the target that can be changed by the pie but not the slice
    min_pro = None

    # X axis, Y axis and labels of X axis
    X = []
    Y = []
    X_lab = []

    # For each slice in slice_LL
    for index in range(len(slice_LL)):
        # If the slice has not been visited or removed yet
        if not index in visited_Dic and (root is None or root not in removed_Dic or index not in removed_Dic[root]):
            # Get the list of target's value that can be changed by the pie but not the slice
            tar_con_pie_not_sli_val_L = get_tar_con_pie_not_sli_val_L(target, tar_con_pie_time_LL, index)

            # If not enough sample
            if len(tar_con_pie_not_sli_val_L) <= sample_size_cutoff:
                continue

            # Get the probalbity of the target that can be changed by the pie but not the slice
            pro_tar_con_pie_not_sli = np.mean(tar_con_pie_not_sli_val_L)

            # Update min_slice and min_pro
            if min_pro is None or min_pro > pro_tar_con_pie_not_sli:
                min_slice = index
                min_pro = pro_tar_con_pie_not_sli

            # Update X axis, Y axis and labels of X axis
            X.append(index)
            Y.append(pro_tar_con_pie_not_sli)
            X_lab.append(slice_LL[index][0][4:])

    # If the pie cannot be expanded anymore
    if min_slice is None:
        return [pie_L, tar_con_pie_time_LL, root, min_slice]

    # Draw the figure
    plt.plot(X, Y, 'ro')
    plt.xticks(X, X_lab)
    plt.xlabel('Slice')
    plt.ylabel('Probability')
    global fig_num
    plt.savefig(fig_dir + 'fig ' + str(fig_num) + ' expand ' + str(decode(pie_L)))
    fig_num += 1
    plt.close()

    # Update tar_con_pie_time_LL
    tar_con_pie_time_LL = get_tar_con_pie_sli_time_LL(target, pie_L, tar_con_pie_time_LL, min_slice)

    # Add min_slice to the pie
    pie_L.append(min_slice)
    # Write pie_L to spamwriter_log
    spamwriter_log.writerow(['expand pie_L' + ': ', decode(pie_L)])
    # Output the pie
    print(['expand pie_L: ', decode(pie_L)])

    # Update root, the first slice added to the pie
    if root is None:
        root = min_slice
        # Write root to spamwriter_log
        spamwriter_log.writerow(['root' + ': ', slice_LL[root]])

    # Update visited_Dic, now min_slice has been visited
    visited_Dic[min_slice] = 1

    return [pie_L, tar_con_pie_time_LL, root, min_slice]


# Get the list of list of timepoints where the target can be changed by the pie and the slice
def get_tar_con_pie_sli_time_LL(target, pie_L, tar_con_pie_time_LL, index):
    # Initialization
    tar_con_pie_sli_time_LL = []

    # If the pie is None or empty
    if pie_L is None or len(pie_L) == 0:
        # Get the list of list of timepoints where the target can be changed by the slice
        tar_con_pie_sli_time_LL = get_tar_con_pie_time_LL(target, [index])
    else:
        # Get the list of list of timepoints where the target can be changed by the slice
        tar_con_sli_time_LL = get_tar_con_pie_time_LL(target, [index])

        # Get sli_time_Dic
        # key: var
        # Val: time
        sli_time_Dic = {}
        for tar_con_sli_time_L in tar_con_sli_time_LL:
            for time in tar_con_sli_time_L:
                sli_time_Dic[time] = 1

        # Get tar_con_pie_sli_time_LL
        for tar_con_pie_time_L in tar_con_pie_time_LL:
            tar_con_pie_sli_time_L = []
            for time in tar_con_pie_time_L:
                if time in sli_time_Dic:
                    tar_con_pie_sli_time_L.append(time)
            if len(tar_con_pie_sli_time_L) > 0:
                tar_con_pie_sli_time_LL.append(tar_con_pie_sli_time_L)

    return tar_con_pie_sli_time_LL


# Check the necessary condition and exclude the slices that are not in the causal pie
def check_nec_con(target, pie_L):
    # Output log file
    spamwriter_log.writerow(["check_nec_con target: ", target])
    spamwriter_log.writerow(["check_nec_con pie_L: ", decode(pie_L)])

    # Backup pie_L
    backup_pie_L = [] + pie_L

    # Check each slice
    for index in backup_pie_L:
        # Get pie \ slice
        temp_L = [] + pie_L
        temp_L.remove(index)

        # Get the list of target's value that can be changed by pie \ slice
        tar_con_pie_time_LL = get_tar_con_pie_time_LL(target, temp_L)

        # Get the list of target's value that can be changed by the slice
        tar_con_sli_time_LL = get_tar_con_pie_time_LL(target, [index])

        # Get the list of list of timepoints where the target can be changed by the pie but not the slice
        tar_con_pie_not_sli_time_LL = get_tar_con_pie_not_sli_time_LL(tar_con_pie_time_LL, tar_con_sli_time_LL)

        # Get the list of target's value that can be changed by the pie but not the slice
        tar_con_pie_not_sli_val_L = get_tar_val_L_time_slot(target, tar_con_pie_not_sli_time_LL)

        # If not enough sample
        if len(tar_con_pie_not_sli_val_L) <= sample_size_cutoff:
            continue

        # Get average length of time slot
        ave_len_time_slot = get_ave_len_time_slot(target, tar_con_pie_not_sli_time_LL)

        # Get the list of target's value relative to the average length of time slot
        tar_val_L = get_tar_val_L_ave_len_time_slot(target, ave_len_time_slot)

        # Unpaired t test
        t, p = stats.ttest_ind(tar_con_pie_not_sli_val_L, tar_val_L, equal_var = False)

        # Output log file
        spamwriter_log.writerow(["check_nec_con slice_LL[index]: ", slice_LL[index]])
        spamwriter_log.writerow(["check_nec_con t: ", t])
        spamwriter_log.writerow(["check_nec_con p: ", p])
        spamwriter_log.writerow(["check_nec_con mean(tar_con_pie_not_sli_val_L): ", np.mean(tar_con_pie_not_sli_val_L)])
        spamwriter_log.writerow(["check_nec_con mean(tar_val_L): ", np.mean(tar_val_L)])
        spamwriter_log.writerow('')

        # If pie \ slice still significantly increases the occurrence of the target
        if t > 0 and p < p_val_cutoff:
            # Remove the slice (since it is not necessary)
            pie_L.remove(index)

    return pie_L


# Remove the influence of the pie from the data
def remove_inf(target, tar_con_pie_time_LL):
    # Remove the influence of the pie from the data
    for tar_con_pie_time_L in tar_con_pie_time_LL:
        for time in tar_con_pie_time_L:
            # If the target was changed by the pie at the current time
            if time in val_Dic[target] and val_Dic[target][time] == 1:
                val_Dic[target][time] = -1


# Shrink the pie by removing the slice that yields the maximum probability of the target that can be changed by the remaining pie but not the slice
def shrink(target, pie_L, root, visited_Dic, removed_Dic):
    # This is the slice that yields the maximum probability of the target that can be changed by the remaining pie but not the slice
    max_slice = None
    # This is the list of slice that yields not enough sample
    max_slice_con_L = []
    # This is the probability of the target that can be changed by the remaining pie but not the slice
    max_pro = None
    # This is the list of list of timepoints where the target can be changed by the remaining pie but not the slice
    max_tar_con_pie_time_LL = []

    # X axis, Y axis and labels of X axis
    X = []
    Y = []
    X_lab = []

    # For each slice in the pie
    for index in pie_L:
        # The max_slice cannot be the root
        if index == root:
            continue

        # Get pie \ slice
        temp_L = [] + pie_L
        temp_L.remove(index)

        # Get the list of list of timepoints where the target can be changed by temp_L (i.e., pie \ slice)
        tar_con_pie_time_LL = get_tar_con_pie_time_LL(target, temp_L)

        # Get the list of target's value that can be changed by temp_L (i.e., pie \ slice) but not the slice
        tar_con_pie_not_sli_val_L = get_tar_con_pie_not_sli_val_L(target, tar_con_pie_time_LL, index)

        # If not enough sample
        if len(tar_con_pie_not_sli_val_L) <= sample_size_cutoff:
            max_slice_con_L.append(index)
            continue

        # Get the probalbity of the target when the slice is absent
        pro_tar_con_pie_not_sli = np.mean(tar_con_pie_not_sli_val_L)

        # Update max_slice and max_pro
        if max_pro is None or max_pro < pro_tar_con_pie_not_sli:
            max_slice = index
            max_pro = pro_tar_con_pie_not_sli
            max_tar_con_pie_time_LL = tar_con_pie_time_LL

        # Update X axis, Y axis and labels of X axis
        X.append(index)
        Y.append(pro_tar_con_pie_not_sli)
        X_lab.append(slice_LL[index][0][4:])

    # If the pie cannot be shrinked anymore
    if max_slice is None and len(max_slice_con_L) == 0:
        return [pie_L, max_tar_con_pie_time_LL, max_slice]

    # Draw the figure
    plt.plot(X, Y, 'ro')
    plt.xticks(X, X_lab)
    plt.xlabel('Slice')
    plt.ylabel('Probability')
    global fig_num
    plt.savefig(fig_dir + 'fig ' + str(fig_num) + ' shrink ' + str(decode(pie_L)))
    fig_num += 1
    plt.close()

    # Clear X axis, Y axis and labels of X axis
    X = []
    Y = []
    X_lab = []

    # Update max_slice_con_L
    if max_slice is not None:
        max_slice_con_L.append(max_slice)

    # Clear max_slice and max_pro
    max_slice = None
    max_pro = None

    # Get the list of list of timepoints where the target can be changed by the root
    tar_con_root_time_LL = get_tar_con_pie_time_LL(target, [root])

    # For each slice in max_slice_con_L
    for index in max_slice_con_L:
        # Get the list of target's value that cannot be changed by the slice
        tar_con_root_not_sli_val_L = get_tar_con_pie_not_sli_val_L(target, tar_con_root_time_LL, index)

        # If not enough sample
        if len(tar_con_root_not_sli_val_L) <= sample_size_cutoff:
            continue

        # Get the probalbity of the target that cannot be changed by the slice
        pro_tar_con_root_not_sli = np.mean(tar_con_root_not_sli_val_L)

        # Update max_slice and max_pro
        if max_pro is None or max_pro < pro_tar_con_root_not_sli:
            max_slice = index
            max_pro = pro_tar_con_root_not_sli

        # Update X axis, Y axis and labels of X axis
        X.append(index)
        Y.append(pro_tar_con_root_not_sli)
        X_lab.append(slice_LL[index][0][4:])

    # If the pie cannot be shrinked anymore
    if max_slice is None:
        return [pie_L, max_tar_con_pie_time_LL, max_slice]

    # Draw the figure
    plt.plot(X, Y, 'ro')
    plt.xticks(X, X_lab)
    plt.xlabel('Slice')
    plt.ylabel('Probability')
    plt.savefig(fig_dir + 'fig ' + str(fig_num) + ' shrink ' + str(decode(pie_L)))
    fig_num += 1
    plt.close()

    # Remove max_slice from the pie
    pie_L.remove(max_slice)
    # Write pie_L to spamwriter_log
    spamwriter_log.writerow(['shrink pie_L' + ': ', decode(pie_L)])
    # Output the pie
    print(['shrink pie_L: ', decode(pie_L)])

    # Update visited_Dic, now min_slice has not been visited
    del visited_Dic[max_slice]

    # Update removed_Dic
    if not root in removed_Dic:
        removed_Dic[root] = {}
    if not max_slice in removed_Dic[root]:
        removed_Dic[root][max_slice] = 1

    # Update max_tar_con_pie_time_LL
    max_tar_con_pie_time_LL = get_tar_con_pie_time_LL(target, pie_L)

    return [pie_L, max_tar_con_pie_time_LL, max_slice]


# Main function
if __name__=="__main__":
    # get parameters from command line
    # please see details of the parameters in the readme file
    src_data_file = sys.argv[1]
    tar_data_file = sys.argv[2]
    pie_file = sys.argv[3]
    log_file = sys.argv[4]
    fig_dir = sys.argv[5]
    pie_size_cutoff = int(sys.argv[6])
    p_val_cutoff = float(sys.argv[7])
    sample_size_cutoff = int(sys.argv[8])
    lag_L = sys.argv[9:]

    # Initialization
    initialization(src_data_file, tar_data_file)

    with open(log_file, 'w') as f:
        # Write the log file
        spamwriter_log = csv.writer(f, delimiter = ' ')
        with open(pie_file, 'w') as f:
            # Write the causal pie file
            spamwriter_pie = csv.writer(f, delimiter = ' ')
            # Search for the causal pies
            search()
