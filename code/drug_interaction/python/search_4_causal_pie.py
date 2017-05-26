

# Please cite the following paper when using the code


# Modules
from __future__ import division
from scipy import stats
import sys
import os
import csv
import numpy as np
import matplotlib.pyplot as plt


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

fig_num = 0


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
        spamwriter_log.writerow(['search ' + target + ': ', target])

        # The list of slices in the pie
        pie_L = []

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
            # The loop continues if:
            #     1) the number of target's value changed by the pie is not smaller than sample size cutoff
            # and 2) the pie is not sufficient
            while len(get_pie_A_not_B_val_L(target, pie_L, None)) > sample_size_cutoff \
                    and check_suff_cond(target, pie_L, spamwriter_log) is False:
                [pie_L, root, min_slice] = expand(target, root, pie_L, visited_Dic, removed_Dic, spamwriter_log)
                # If the pie cannot be expanded anymore
                if min_slice is None:
                    break

            # Check if the number of target's value changed by the pie is not smaller than sample size cutoff
            if len(get_pie_A_not_B_val_L(target, pie_L, None)) > sample_size_cutoff:
                # If the pie is sufficient (to produce the effect)
                if check_suff_cond(target, pie_L, spamwriter_log) is True:
                    # Mark each slice in the pie as unvisited (i.e., deleting the key from the dict), except for the root
                    for index in pie_L:
                        if index != root:
                            del visited_Dic[index]

                    # Check the necessary condition (to produce the effect) and remove unnecessary slices
                    pie_L = check_nece_cond(target, pie_L, spamwriter_log)

                    # Mark each slice in the pie as visited (i.e, adding the key to the dict)
                    for index in pie_L:
                        visited_Dic[index] = 1

                    # Remove the influence of the pie from the data
                    remove_inf(target, pie_L)

                    # Write the pie to spamwriter_pie
                    spamwriter_pie.writerow(['causal pie of ' + target + ': ', decode(pie_L)])
                    # Output the pie
                    print(['causal pie of ' + target + ': ', decode(pie_L)])
                else:
                    # Mark each slice in the pie as unvisited (i.e., deleting the key from the dict), except for the root
                    for index in pie_L:
                        if index != root:
                            del visited_Dic[index]
                # Clear
                pie_L = []
                root = None
            else:
                # Shrink
                [pie_L, min_slice] = shrink(target, root, pie_L, visited_Dic, removed_Dic, spamwriter_log)

                # If the pie cannot be shrinked anymore
                if min_slice is None:
                    # Mark each slice in the pie as unvisited (i.e., deleting the key from the dict), except for the root
                    for index in pie_L:
                        if index != root:
                            del visited_Dic[index]

                    # Clear
                    pie_L = []
                    root = None


# Check sufficient condition, i.e., P(target | pie) >> P(target)
def check_suff_cond(target, pie_L, spamwriter_log):
    # Output log file
    spamwriter_log.writerow(["check_suff_cond target: ", target])
    spamwriter_log.writerow(["check_suff_cond pie_L: ", decode(pie_L)])

    # Get sample with respect to [pie \wedge \neg slice] (i.e. the pie excluding the slice)
    val_exc_L = get_pie_A_not_B_val_L(target, pie_L, None)

    min_win_len = get_min_win_len(pie_L)
    val_L = get_val_L_min_win_len(target, min_win_len)

    # Unpaired t test
    t, p = stats.ttest_ind(val_exc_L, val_L, equal_var = False)

    # Output log file
    spamwriter_log.writerow(["check_suff_cond t: ", t])
    spamwriter_log.writerow(["check_suff_cond p: ", p])
    spamwriter_log.writerow(["check_suff_cond mean(val_exc_L): ", np.mean(val_exc_L)])
    spamwriter_log.writerow(["check_suff_cond mean(val_L): ", np.mean(val_L)])
    spamwriter_log.writerow('')

    # If pie does not significantly increase the occurrence of the target
    if t <= 0 or p >= p_val_cutoff:
        return False

    # Check each slice that does not belong to the pie
    for index in range(len(slice_LL)):
        if index in pie_L:
            continue

        # Get sample with respect to [pie \wedge \neg slice] (i.e. the pie excluding the slice)
        val_exc_L = get_pie_A_not_B_val_L(target, pie_L, [index])

        # If the sample size is not larger than the sample size cutoff
        if len(val_exc_L) <= sample_size_cutoff:
            continue

        min_win_len = get_min_win_len(pie_L)
        val_L = get_val_L_min_win_len(target, min_win_len)

        # Unpaired t test
        t, p = stats.ttest_ind(val_exc_L, val_L, equal_var = False)

        # Output log file
        spamwriter_log.writerow(["check_suff_cond slice_LL[index]: ", slice_LL[index]])
        spamwriter_log.writerow(["check_suff_cond t: ", t])
        spamwriter_log.writerow(["check_suff_cond p: ", p])
        spamwriter_log.writerow(["check_suff_cond mean(val_exc_L): ", np.mean(val_exc_L)])
        spamwriter_log.writerow(["check_suff_cond mean(val_L): ", np.mean(val_L)])
        spamwriter_log.writerow('')

        # If pie \ slice does not significantly increase the occurrence of the target
        if t <= 0 or p >= p_val_cutoff:
            return False

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

    # If the pie is empty, return the minimum window length, 1
    if pie_L is None or len(pie_L) == 0:
        return 1

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
        # If the length of the list is not narrower than the minimum length
        else:
            # If temp_L does not contain removed value of the target
            if min(temp_L) != -1:
                # Add the maximum value in the list (so that if the target occurs in the window, it counts as occurred in the window)
                val_L.append(max(temp_L))
            # Reset the list
            temp_L = []

    # If the list is not empty
    if len(temp_L) > 0:
        # If temp_L does not contain removed value of the target
        if min(temp_L) != -1:
            # Add the maximum value in the list (so that if the target occurs in the window, it counts as occurred in the window)
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
            val_L.append(max(temp_L))

    return val_L


# Expand the pie by adding the slice that yields the maximum cooccurrence with the pie
def expand(target, root, pie_L, visited_Dic, removed_Dic, spamwriter_log):
    # This is the slice that yields the minimum probability of the target when the slice is absent
    min_slice = None
    # This is the probability of the target when the slice is absent
    min_pro = None

    # X axis, Y axis and labels of X axis
    X = []
    Y = []
    X_lab = []

    # For each slice in slice_LL
    for index in range(len(slice_LL)):
        # If the slice has not been visited or removed yet
        if not index in visited_Dic and (root is None or root not in removed_Dic or index not in removed_Dic[root]):
            # Get the list of target's value not changed by the current slice
            pie_A_not_B_val_L = get_pie_A_not_B_val_L(target, pie_L, [index])
            # Get the probalbity of the target when the current slice is absent
            pro_pie_A_not_B = np.mean(pie_A_not_B_val_L)

            # Update min_slice and min_pro
            if min_pro is None or min_pro > pro_pie_A_not_B:
                min_slice = index
                min_pro = pro_pie_A_not_B

            # Update X axis, Y axis and labels of X axis
            X.append(index)
            Y.append(pro_pie_A_not_B)
            X_lab.append(slice_LL[index][0][4:])

    # If the pie cannot be expanded anymore
    if min_slice is None:
        return [pie_L, root, min_slice]

    # Draw the figure
    plt.plot(X, Y, 'ro')
    plt.xticks(X, X_lab)
    plt.xlabel('Slice')
    plt.ylabel('Probability')
    global fig_num
    plt.savefig(figure + 'fig ' + str(fig_num) + ' expand ' + str(decode(pie_L)))
    fig_num += 1
    plt.close()

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

    # Write len(cooccur_time_L) to spamwriter_log
    spamwriter_log.writerow(['expand len(get_pie_A_not_B_val_L(target, pie_L, None))' + ': ', len(get_pie_A_not_B_val_L(target, pie_L, None))])

    return [pie_L, root, min_slice]


# Shrink the pie by removing the slice that yields the maximum probability of the target when the slice is absent
def shrink(target, root, pie_L, visited_Dic, removed_Dic, spamwriter_log):
    # This is the slice that yields the maximum probability of the target when the slice is absent
    max_slice = None
    # This is the probability of the target when the slice is absent
    max_pro = None

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

        # Get the list of target's value not changed by slice_LL[index]
        pie_A_not_B_val_L = get_pie_A_not_B_val_L(target, temp_L, [index])
        # Get the probalbity of the target when the slice is absent
        pro_pie_A_not_B = np.mean(pie_A_not_B_val_L)

        # Update max_slice and max_pro
        if max_pro is None or max_pro < pro_pie_A_not_B:
            max_slice = index
            max_pro = pro_pie_A_not_B

        # Update X axis, Y axis and labels of X axis
        X.append(index)
        Y.append(pro_pie_A_not_B)
        X_lab.append(slice_LL[index][0][4:])

    # If the pie cannot be shrinked anymore
    if max_slice is None:
        return [pie_L, max_slice]

    # Draw the figure
    plt.plot(X, Y, 'ro')
    plt.xticks(X, X_lab)
    plt.xlabel('Slice')
    plt.ylabel('Probability')
    global fig_num
    plt.savefig(figure + 'fig ' + str(fig_num) + ' shrink ' + str(decode(pie_L)))
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

    return [pie_L, max_slice]


# Check the necessary condition and exclude the slices that are not in the causal pie
def check_nece_cond(target, pie_L, spamwriter_log):
    # Backup pie_L
    backup_pie_L = [] + pie_L

    # Output log file
    spamwriter_log.writerow(["check_nece_cond target: ", target])
    spamwriter_log.writerow(["check_nece_cond pie_L: ", decode(pie_L)])

    # Check each slice
    for index in backup_pie_L:
        # Get pie \ slice
        temp_L = [] + pie_L
        temp_L.remove(index)

        # Get sample with respect to [pie \ slice \wedge \neg slice] (i.e. the pie \ slice excluding the slice)
        val_exc_L = get_pie_A_not_B_val_L(target, temp_L, [index])

        # If the sample size is not larger than the sample size cutoff
        if len(val_exc_L) <= sample_size_cutoff:
            continue

        min_win_len = get_min_win_len(pie_L)
        val_L = get_val_L_min_win_len(target, min_win_len)

        # Unpaired t test
        t, p = stats.ttest_ind(val_exc_L, val_L, equal_var = False)

        # Output log file
        spamwriter_log.writerow(["check_nece_cond slice_LL[index]: ", slice_LL[index]])
        spamwriter_log.writerow(["check_nece_cond t: ", t])
        spamwriter_log.writerow(["check_nece_cond p: ", p])
        spamwriter_log.writerow(["check_nece_cond mean(val_trg_L): ", np.mean(val_L)])
        spamwriter_log.writerow(["check_nece_cond mean(val_exc_L): ", np.mean(val_exc_L)])
        spamwriter_log.writerow('')

        # If pie \ slice still significantly increases the occurrence of the target
        if t > 0 and p < p_val_cutoff:
            # Remove the slice (since it is not necessary)
            pie_L.remove(index)

    return pie_L


# Remove the influence of the pie from the data
def remove_inf(target, pie_L):
    # Get the list of timepoints where the target can be changed by the pie
    pie_time_LL = get_pie_time_LL(pie_L)

    # Remove the influence of the pie from the data
    for pie_time_L in pie_time_LL:
        for time in pie_time_L:
            # If the target was changed by the pie and the current time
            if time in val_Dic[target] and val_Dic[target][time] == 1:
                val_Dic[target][time] = -1


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
    figure = sys.argv[5]
    pie_size_cutoff = int(sys.argv[6])
    p_val_cutoff = float(sys.argv[7])
    sample_size_cutoff = int(sys.argv[8])
    lag_L = sys.argv[9:]

    # Make directory
    directory = os.path.dirname(pie_file)
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = os.path.dirname(log_file)
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = os.path.dirname(figure)
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
