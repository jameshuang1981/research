

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
def search(spamwriter_log, spamwriter_pie):
    for target in trg_Dic:
        # Write target to spamwriter_log
        spamwriter_log.writerow([target + ': ', target])

        # The list of timepoints where the target occurs
        time_L = []
        for time in val_Dic[target]:
            if val_Dic[target][time] == 1:
                time_L.append(time)

        # The dictionary records the visited slice
        visited_Dic = {}

        # The dictionary records the removed slice relative to the root
        removed_Dic = {}

        # The list of slices in the pie
        pie_L = []

        # The first slice added to the pie
        root = None

        # The loop continues if there are unvisited nodes
        while len(visited_Dic) < len(slice_LL):
            # If the pie is empty
            if len(pie_L) == 0:
                # Initialization
                cooccur_time_L = time_L

            # The loop continues if:
            #     1) the number of timepoints in cooccur_time_L is not smaller than sample size cutoff
            # and 2) the pie includes at least two slices, or the pie is not sufficient
            while len(cooccur_time_L) >= sample_size_cutoff and (len(pie_L) < 2 or check_sff_cnd(target, pie_L, spamwriter_log) is False):
                [pie_L, cooccur_time_L, root, max_slice] = expand(cooccur_time_L, root, pie_L, visited_Dic, removed_Dic, spamwriter_log)
                # If the pie cannot be expanded anymore
                if max_slice is None:
                    break

            # Check if the number of timepoints in cooccur_time_L is not smaller than sample size cutoff
            if len(cooccur_time_L) >= sample_size_cutoff:
                print('hi')
                # If the pie is sufficient
                if check_sff_cnd(target, pie_L, spamwriter_log) is True:
                    print('morning')
                    # Mark each slice in the pie as unvisited (i.e., deleting the key from the dict), except for the root
                    for index in pie_L:
                        if index != root:
                            del visited_Dic[index]

                    # Check the necessary condition and remove unnecessary slices
                    pie_L = check_ncs_cnd(target, pie_L, spamwriter_log)

                    # Mark each slice in the pie as visited (i.e, adding the key to the dict)
                    for index in pie_L:
                        visited_Dic[index] = 1

                    # Write the pie to spamwriter_pie
                    spamwriter_pie.writerow([target + ': ', decode(pie_L)])
                    # Output the pie
                    print(decode(pie_L))
                else:
                    print(root)
                    # Mark each slice in the pie as unvisited (i.e., deleting the key from the dict), except for the root
                    for index in pie_L:
                        if index != root:
                            del visited_Dic[index]
                # Clear
                pie_L = []
                root = None
            else:
                # Shrink
                [pie_L, cooccur_time_L] = shrink(time_L, root, pie_L, visited_Dic, removed_Dic, spamwriter_log)


# Check sufficient condition, i.e., P(target | pie) >> P(target)
def check_sff_cnd(target, pie_L, spamwriter_log):
    if len(pie_L) == 0:
        return False

    # Write log file
    spamwriter_log.writerow(["check_sff_cnd: ", target])
    spamwriter_log.writerow(["target: ", target])
    spamwriter_log.writerow(["pie_L: ", decode(pie_L)])

    # Get val_trg_cnd_pie_L
    val_trg_cnd_pie_L = get_pie_A_not_B_val_L(target, pie_L, None)

    # Get the minimum window length of slices in the pie
    min_win_len = get_min_win_len(pie_L)
    val_L = get_val_L_min_win_len(target, min_win_len)
    # print(val_L)

    # print(['1:', val_trg_cnd_pie_L])
    # print(['2:', val_trg_L])

    # Unpaired t test
    t, p = stats.ttest_ind(val_trg_cnd_pie_L, val_L, equal_var=False)

    # Output log file
    spamwriter_log.writerow(["t: ", t])
    spamwriter_log.writerow(["p: ", p])
    spamwriter_log.writerow(["mean(val_trg_cnd_pie_L): ", np.mean(val_trg_cnd_pie_L)])
    spamwriter_log.writerow(["mean(val_L): ", np.mean(val_L)])
    spamwriter_log.writerow('')

    # If the pie does not significantly increase the occurrence of the target
    if t <= 0 or p >= p_val_cutoff:
        return False
    else:
        return True


# Get the list of value of the target due to pie A but not pie B
def get_pie_A_not_B_val_L(target, pie_A_L, pie_B_L):
    # Get the list of timepoints where the target can be changed by pie A
    # Initialization
    pie_A_time_LL = []
    # If there is no pie A, the list includes all timepoints
    if pie_A_L is None or len(pie_A_L) == 0:
        for time in val_Dic[target]:
            pie_A_time_LL.append([time])
    # Otherwise, the list only includes timepoints where the target can be changed by pie A
    else:
        pie_A_time_LL = get_pie_time_LL(pie_A_L)

    # Get the list of timepoints where the target can be changed by pie B
    pie_B_time_LL = get_pie_time_LL(pie_B_L)

    # Get the list of timepoints where the target can be changed by pie A but not pie B
    pie_A_not_B_time_LL = get_pie_A_not_B_time_LL(pie_A_time_LL, pie_B_time_LL)

    # Get the list of value of the target due to pie A but not pie B
    val_L = get_val_L(target, pie_A_not_B_time_LL)

    return val_L


# Get the list of timepoints where the target can be changed by the pie
def get_pie_time_LL(pie_L):
    # Initialization
    pie_time_LL = []

    # If the pie is None or empty, return empty list
    if pie_L is None or len(pie_L) == 0:
        return pie_time_LL

    # Get the minimum window length of slices in the pie
    min_win_len = get_min_win_len(pie_L)

    # Get dictionary of start and end
    [start_Dic, end_Dic] = get_start_end_Dic(pie_L)

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
        if len(pie_time_Dic) == len(pie_L):
            if recorded_F is False:
                time_L = []
                recorded_F = True
            time_L.append(time)
            # If the last timepoint or the length of the intersection equals the minimum window length
            if time == max_time_stamp or len(time_L) == min_win_len:
                pie_time_LL.append(time_L)
                recorded_F = False
        # If some slices are absent and we have been recording time
        elif recorded_F:
            pie_time_LL.append(time_L)
            recorded_F = False

    return pie_time_LL


# Get the minimum window length of slices in the pie
def get_min_win_len(pie_L):
    # Initialization
    min_win_len = None

    # If the pie is empty, return the default value (None)
    if pie_L is None or len(pie_L) == 0:
        return min_win_len

    # For each slice in the pie
    for index in pie_L:
        # Get window start, window end, and the length
        win_start = slice_LL[index][1]
        win_end = slice_LL[index][2]
        len_intersection = win_end - win_start + 1
        # Update the minimum length
        if min_win_len is None or min_win_len > len_intersection:
            min_win_len = len_intersection

    return min_win_len


# Get the list of value in windows no wider than the minimum length
def get_val_L_min_win_len(target, min_win_len):
    # Initialization
    # val_L is the return value (i.e. the list of value)
    val_L = []
    # temp_L is the list of value in windows no wider than the minimum length
    temp_L = []

    # For each timepoint where the target is measured
    for time in val_Dic[target]:
        # If the length of the list is still narrower than the minimum length, add the value
        if len(temp_L) < min_win_len:
            temp_L.append(val_Dic[target][time])
        # If the length of the list is not narrower than the minimum length, add the maximum value in the list
        # (so that if the target occurs in the window, it counts as occurred in the window)
        else:
            val_L.append(max(temp_L))
            # Reset the list
            temp_L = []

    # If the list is not empty, add the maximum value in the list
    if len(temp_L) > 0:
        val_L.append(max(temp_L))

    return val_L


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


# Get the timepoints when pie A is present whereas pie B is absent
def get_pie_A_not_B_time_LL(pie_A_time_LL, pie_B_time_LL):
    if pie_B_time_LL is None or len(pie_B_time_LL) == 0:
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

    if time_LL is None or len(time_LL) == 0:
        return val_L

    # For each time_L, get the maximum absolute value
    for time_L in time_LL:
        # Initialization
        max_abs_val = None
        for time in time_L:
            if time in val_Dic[target] and (max_abs_val is None or abs(max_abs_val) < abs(val_Dic[target][time])):
                max_abs_val = val_Dic[target][time]
        if max_abs_val is not None:
            val_L.append(max_abs_val)

    return val_L


# Get the list of timepoints (from time_L) where the slice occurs
def get_cooccur_time_L(time_L, slice_L):
    # Initialization
    cooccur_time_L = []

    for time in time_L:
        # Decompose the slice into var, window start, and window end
        [var, win_start, win_end] = slice_L
        # For each previous time in [time - window end, time - window start]
        for prev_time in range(time - win_end, time - win_start + 1):
            # If the slice occurs at previous time
            if prev_time in val_Dic[var] and val_Dic[var][prev_time] == 1:
                # Add the time to the list of timepoints
                cooccur_time_L.append(time)
                break

    return cooccur_time_L


# Expand the pie by adding the slice that yields the maximum cooccurrence with the pie
def expand(cooccur_time_L, root, pie_L, visited_Dic, removed_Dic, spamwriter_log):
    # This is the slice that yields the maximum cooccurrence with the pie
    max_slice = None
    # This is the list of timepoints where max_slice and pie_L cooccur
    max_time_L = None

    # For each slice in slice_LL
    for index in range(len(slice_LL)):
        # If the slice has not been visited or removed yet
        if not index in visited_Dic and (root is None or root not in removed_Dic or index not in removed_Dic[root]):
            # Get the list of timepoints where slice_LL[index] and pie_L cooccur
            temp_time_L = get_cooccur_time_L(cooccur_time_L, slice_LL[index])

            # Update max_slice, max_time_L and max_val_L
            if max_time_L is None or len(max_time_L) < len(temp_time_L):
                max_slice = index
                max_time_L = temp_time_L

    if max_slice is None:
        return [pie_L, cooccur_time_L, root, max_slice]

    # Add max_slice to the pie
    pie_L.append(max_slice)
    # Write pie_L to spamwriter_log
    spamwriter_log.writerow(['pie_L' + ': ', decode(pie_L)])
    # Output the pie
    print(['expand pie_L: ', decode(pie_L)])

    # Update root, the first slice added to the pie
    if root is None:
        root = max_slice
        # Write root to spamwriter_log
        spamwriter_log.writerow(['root' + ': ', slice_LL[root]])

    # Update visited_Dic, now max_slice has been visited
    visited_Dic[max_slice] = 1

    # Update cooccur_time_L (with fewer timepoints since a new slice, max_slices, has been added to pie_L)
    cooccur_time_L = max_time_L
    # Write len(cooccur_time_L) to spamwriter_log
    spamwriter_log.writerow(['len(cooccur_time_L)' + ': ', len(cooccur_time_L)])

    return [pie_L, cooccur_time_L, root, max_slice]


# Shrink the pie by removing the slice that yields the maximum cooccurrence of the other slices in the pie
def shrink(time_L, root, pie_L, visited_Dic, removed_Dic, spamwriter_log):
    # This is the slice that yields the maximum cooccurrence of the other slices in the pie
    max_slice = None
    # This is the list of timepoints where the other slices other than max_slice cooccur
    max_time_L = None

    # For each slice in the pie
    for index in pie_L:
        # The max_slice cannot be the root
        if index == root:
            continue
        # Get the list of timepoints where the other slices other than the current slice cooccur
        # Initialization
        temp_time_L = time_L
        for other_index in pie_L:
            if not other_index == index:
                temp_time_L = get_cooccur_time_L(temp_time_L, slice_LL[other_index])

        # Update max_slice and max_time_L
        if max_time_L is None or len(max_time_L) < len(temp_time_L):
            max_slice = index
            max_time_L = temp_time_L

    # Remove max_slice from the pie
    pie_L.remove(max_slice)
    # Write pie_L to spamwriter_log
    spamwriter_log.writerow(['pie_L' + ': ', decode(pie_L)])
    # Output the pie
    print(['shrink pie_L: ', decode(pie_L)])

    # Update visited_Dic, now max_slice has not been visited
    del visited_Dic[max_slice]

    # Update removed_Dic
    if not root in removed_Dic:
        removed_Dic[root] = {}
    if not max_slice in removed_Dic[root]:
        removed_Dic[root][max_slice] = 1

    return [pie_L, max_time_L]


# Check the necessary condition and exclude the slices that are not in the causal pie
def check_ncs_cnd(target, pie_L, spamwriter_log):
    # Backup pie_L
    backup_pie_L = [] + pie_L

    # Check each slice
    for index in backup_pie_L:
        # Get pie \ slice
        temp = [] + pie_L
        temp.remove(index)
        if len(temp) == 0:
            return temp

        # Output log file
        spamwriter_log.writerow(["target: ", target])
        spamwriter_log.writerow(["pie_L: ", decode(pie_L)])

        # Get sample with respect to [pie \ slice \wedge \neg slice] (i.e. the pie \ slice excluding the slice)
        val_exc_L = get_pie_A_not_B_val_L(target, temp, [index])

        min_win_len = get_min_win_len(pie_L)
        val_L = get_val_L_min_win_len(target, min_win_len)

        # Unpaired t test
        t, p = stats.ttest_ind(val_exc_L, val_L, equal_var = False)

        # Output log file
        spamwriter_log.writerow(["t: ", t])
        spamwriter_log.writerow(["p: ", p])
        spamwriter_log.writerow(["mean(val_trg_L): ", np.mean(val_L)])
        spamwriter_log.writerow(["mean(val_exc_L): ", np.mean(val_exc_L)])
        spamwriter_log.writerow('')

        # If the pie does not significantly increase the occurrence of the target
        if t > 0 and p < p_val_cutoff:
            pie_L.remove(index)

    return pie_L


# Get the actual slices in the pie
def decode(pie_L):
    temp_L = []
    if pie_L:
        for index in pie_L:
            temp_L.append(slice_LL[index])

    return temp_L


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
        # Write the log file
        spamwriter_log = csv.writer(f, delimiter = ' ')
        with open(pie_file, 'w') as f:
            # Write the causal pie file
            spamwriter_pie = csv.writer(f, delimiter = ' ')
            # Search for the causal pies
            search(spamwriter_log, spamwriter_pie)
