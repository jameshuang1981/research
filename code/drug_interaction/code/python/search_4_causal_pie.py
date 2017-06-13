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
import math

# Notations
# _L      : indicates the data structure is a list
# _LL     : indicates the data structure is a list of list
# _Dic    : indicates the data structure is a dictionary
# _L_Dic  : indicates the data structure is a dictionary, where the value is a list
# _LL_Dic : indicates the data structure is a dictionary, where the value is a list of list


# Global variables
# The list of time windows, where each window, win, is a list, [win_start, win_end]
win_LL = []

# The list of timepoints
time_series_L = []

# The list of slices
slice_LL = []

# The dictionary of sources
# key: var
# val: 1
src_Dic = {}

# The dictionary of targets
# key: var
# val: 1
tar_Dic = {}

# The dictionary of time series
# key: time
# val: 1
time_series_Dic = {}

# The dictionary of value
# key: var->time
# val: value of var at the time
val_Dic = {}

# The dictionary of var
# key: time
# val: the vars occur at the time
var_Dic = {}

# The dictionary of P(target)
# key: target
# val: P(target)
pro_Dic = {}

# The dictionary of 1 - (target)
# key: target
# val: 1 - P(target)
not_pro_Dic = {}

# The dictionary of #(target), that is the number of timepoints where the target is measured
# key: target
# val: #(target)
num_Dic = {}

# The dictionary of #(target = 1), that is the number of timepoints where the target is 1
# key: target
# val: #(target = 1)
num_1_Dic = {}

# The dictionary records the removed slice
# key: slice
# val: 1
removed_Dic = {}

# The dictionary records the replaced slice relative to the root
# key: slice
# val: 1
replaced_Dic = {}

# The dictionary records the list of pies for which the slice was conditioned to check the sufficiency condition
# key: slice
# val: list of pies
conditioned_Dic = {}

# The dictionary of pies
# key: target
# val: list of pies
pie_Dic = {}

# The maximum time stamp
max_time_stamp = 0

# The minimum size of the samples
sample_size_cutoff = 30

# The number of the figures
fig_num = 0


# Initialization
# @param        src_data_file           source data file, which includes variables that can be the causes, the data are of the following form
#                                  var1 (i.e. header), var1_t1 (i.e. val), ..., var1_tn (i.e. val)
#                                  , ...,
#                                  varn (i.e. header), varn_t1 (i.e. val), ..., varn_tn (i.e. val)
# @param        tar_data_file           target data file, which includes variables that can be the effects, the data are of the following form
#                                  var1 (i.e. header), var1_t1 (i.e. val), ..., var1_tn (i.e. val)
#                                  , ...,
#                                  varn (i.e. header), varn_t1 (i.e. val), ..., varn_tn (i.e. val)
def initialization(src_data_file, tar_data_file):
    # Load source file
    load_data(src_data_file, True)

    # Load target file
    load_data(tar_data_file, False)

    # Get windows
    get_win_LL(lag_L)

    # Get slices
    get_slice_LL()

    # Get time series
    get_time_series()

    # Get the statistics (pro_Dic, not_pro_Dic, num_Dic, and num_1_Dic) of the targets
    get_tar_statistics()

    # Get max time stamp
    global max_time_stamp
    max_time_stamp = time_series_L[len(time_series_L) - 1]


# Load data, get data_type_Dic, val_Dic, src_Dic and tar_Dic
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
        spamreader = list(csv.reader(f, delimiter=','))

        # Get data_type_Dic, val_Dic, src_Dic and tar_Dic
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
                        # Get tar_Dic
                        if not var in tar_Dic:
                            tar_Dic[var] = 1


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


# Get the statistics (pro_Dic, not_pro_Dic, num_Dic, and num_1_Dic) of the targets
def get_tar_statistics():
    for target in tar_Dic:
        val_L = []
        for time in val_Dic[target]:
            val_L.append(val_Dic[target][time])

        pro_Dic[target] = np.mean(val_L)
        not_pro_Dic[target] = 1 - pro_Dic[target]
        num_Dic[target] = len(val_L)
        num_1_Dic[target] = sum(val_L)


# Search for the causal pies
def search():
    for target in tar_Dic:
        # Write target to spamwriter_log
        spamwriter_log.writerow(['search ' + target + ': ', target])
        f_log.flush()

        # The list of slices in the pie
        pie_L = []

        # The list of list of timepoints where the target can be changed by the pie
        # Initialization
        tar_con_pie_time_LL = get_tar_con_pie_time_LL(target, pie_L)

        # The first slice added to the pie
        root = None

        # The dictionary records the removed slice
        # key: slice
        # val: 1
        global removed_Dic
        removed_Dic = {}

        # The dictionary records the replaced slice
        # key: slice
        # val: 1
        global replaced_Dic
        replaced_Dic = {}

        # The dictionary records the list of pies for which the slice was conditioned to check the sufficiency condition
        # key: slice
        # val: list of pies
        global conditioned_Dic
        conditioned_Dic = {}

        # The loop continues if there are unvisited nodes
        while len(removed_Dic) < len(slice_LL):
            # Flag eno_sam_F, indicating whether there is enough sample
            # Flag suf_F, indicating whether the pie is sufficient
            pie_L, tar_con_pie_time_LL, eno_sam_F, suf_F = check_suf_con(target, pie_L, tar_con_pie_time_LL)

            # If there is enough sample
            if eno_sam_F is True:
                # The loop continues if the pie is not sufficient (to produce the effect)
                while suf_F is False:
                    [pie_L, tar_con_pie_time_LL, root, min_slice] = expand(target, pie_L, tar_con_pie_time_LL, root)
                    # If the pie cannot be expanded anymore
                    if min_slice is None:
                        break
                    # Flag eno_sam_F, indicating whether there is enough sample
                    # Flag suf_F, indicating whether the pie is sufficient
                    pie_L, tar_con_pie_time_LL, eno_sam_F, suf_F = check_suf_con(target, pie_L, tar_con_pie_time_LL)

            # If the pie is sufficient
            if suf_F is True:
                # Check the necessary condition (to produce the effect) and remove unnecessary slices
                # Flag removed_F, indicating whether there are removed necessary slices
                pie_L, removed_F = check_nec_con(target, pie_L)

                # If all the slices in the pie are necessary
                if removed_F is False:
                    # Check whether the pie has been found
                    if not target in pie_Dic or not is_in(pie_L, pie_Dic[target]):
                        # Update removed_Dic
                        # Mark each slice in the pie as removed (i.e, adding the key to the dict)
                        for index in pie_L:
                            removed_Dic[index] = 1

                        # Remove the influence of the pie from the data
                        remove_inf(target, tar_con_pie_time_LL)

                        # Write the pie to spamwriter_pie
                        spamwriter_pie.writerow(['causal pie of ' + target + ': ', decode(pie_L)])
                        f_pie.flush()
                        # Output the pie
                        print(['causal pie of ' + target + ': ', decode(pie_L)])

                    # Mark the root as removed (i.e, adding the key to the dict)
                    removed_Dic[root] = 1

                    # Clear
                    pie_L = []
                    tar_con_pie_time_LL = get_tar_con_pie_time_LL(target, pie_L)
                    root = None
                    replaced_Dic = {}
                else:
                    # Update
                    tar_con_pie_time_LL = get_tar_con_pie_time_LL(target, pie_L)

            else:
                # Shrink
                [pie_L, tar_con_pie_time_LL, max_slice] = shrink(target, pie_L)

                # If the pie cannot be shrinked anymore
                if max_slice is None:
                    # Update removed_Dic
                    # Mark the root as removed (i.e, adding the key to the dict)
                    removed_Dic[root] = 1

                    # Clear
                    pie_L = []
                    tar_con_pie_time_LL = get_tar_con_pie_time_LL(target, pie_L)
                    root = None
                    replaced_Dic = {}


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
            # If the target is measured at this time
            if time in val_Dic[target]:
                time_L.append(time)
            # If the last timepoint or the length of the intersection equals the minimum window length
            if time == max_time_stamp or len(time_L) == min_win_len:
                tar_con_pie_time_LL.append(time_L)
                recorded_F = False
        # If some slices are absent and we have been recording time
        elif recorded_F:
            if len(time_L) > 0:
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
    f_log.flush()

    # Flag, indicating whether there is enough sample, True by default
    eno_sam_F = True
    # Flag, indicating whether the pie is sufficient, False by default
    suf_F = False

    # If the pie is None or empty
    if pie_L is None or len(pie_L) == 0:
        return [pie_L, tar_con_pie_time_LL, eno_sam_F, suf_F]

    # Get P(target | pie)
    pro_tar_con_pie, num_tar_con_pie, num_tar_1_con_pie = get_pro_num_tar_con_pie(target, tar_con_pie_time_LL)
    spamwriter_log.writerow(["check_suf_con pro_tar_con_pie: ", pro_tar_con_pie])
    spamwriter_log.writerow(["check_suf_con num_tar_con_pie: ", num_tar_con_pie])
    spamwriter_log.writerow(["check_suf_con num_tar_1_con_pie: ", num_tar_1_con_pie])
    f_log.flush()

    # If P(target | pie) is None or not enough sample
    if pro_tar_con_pie is None or num_tar_con_pie <= sample_size_cutoff:
        # Update eno_sam_F, since there is no enough sample
        eno_sam_F = False
        return [pie_L, tar_con_pie_time_LL, eno_sam_F, suf_F]

    # Get numerator
    pro_tar = pro_Dic[target]
    numerator = pro_tar_con_pie - pro_tar
    spamwriter_log.writerow(["check_suf_con numerator: ", numerator])
    f_log.flush()

    # Get denominator
    num_tar = num_Dic[target]
    num_tar_1 = num_1_Dic[target]
    pro = (num_tar_1_con_pie + num_tar_1) / (num_tar_con_pie + num_tar)
    denominator = math.sqrt(pro * (1 - pro) * (1 / num_tar_con_pie + 1 / num_tar))

    # If denominator is zero
    if denominator == 0:
        return [pie_L, tar_con_pie_time_LL, eno_sam_F, suf_F]

    # Get z value
    z_val = numerator / denominator
    # Get p value
    p_val = stats.norm.sf(z_val)

    # Output log file
    spamwriter_log.writerow(["check_suf_con z_val: ", z_val])
    spamwriter_log.writerow(["check_suf_con p_val: ", p_val])
    spamwriter_log.writerow('')
    f_log.flush()

    # If the pie does not significantly increase the occurrence of the target
    if p_val >= p_val_cutoff_suf:
        return [pie_L, tar_con_pie_time_LL, eno_sam_F, suf_F]

    # The sum of vote of the slices, 0 by default
    vote_sum = 0

    # Check each slice that has not been used for checking the sufficiency for the current pie
    for index in range(len(slice_LL)):
        # Get pie_vote_F_L
        pie_vote_F_L = get_pie_vote_F_L(pie_L, index)
        if pie_vote_F_L is not None:
            # Get the vote of the slice
            vote_F =  pie_vote_F_L[1]
            # Update vote_sum
            vote_sum += vote_F
            continue

        # Output log file
        spamwriter_log.writerow(["check_suf_con slice_LL[index]: ", slice_LL[index]])
        f_log.flush()

        # Get the list of list of timepoints where the target can be changed by the slice
        tar_con_sli_time_LL = get_tar_con_pie_time_LL(target, [index])

        # Get the list of list of timepoints where the target can be changed by the pie but not the slice
        tar_con_pie_not_sli_time_LL = get_tar_con_pie_not_sli_time_LL(tar_con_pie_time_LL, tar_con_sli_time_LL)

        # Get P(target | pie \ slice)
        pro_tar_con_pie_not_sli, num_tar_con_pie_not_sli, num_tar_1_con_pie_not_sli = get_pro_num_tar_con_pie(target,
                                                                                                              tar_con_pie_not_sli_time_LL)
        spamwriter_log.writerow(["check_suf_con pro_tar_con_pie_not_sli: ", pro_tar_con_pie_not_sli])
        spamwriter_log.writerow(["check_suf_con num_tar_con_pie_not_sli: ", num_tar_con_pie_not_sli])
        spamwriter_log.writerow(["check_suf_con num_tar_1_con_pie_not_sli: ", num_tar_1_con_pie_not_sli])
        f_log.flush()

        # Initialize conditioned_Dic
        if not index in conditioned_Dic:
            conditioned_Dic[index] = []

        # Initialize the vote of the slice, 0 by default
        vote_F = 0

        # If P(target | pie) is None
        if pro_tar_con_pie_not_sli is None:
            # Update conditioned_Dic
            conditioned_Dic[index].append([list(pie_L), vote_F])
            continue

        # Get numerator
        numerator = pro_tar_con_pie_not_sli - pro_tar
        spamwriter_log.writerow(["check_suf_con numerator: ", numerator])
        f_log.flush()

        # Get denominator
        pro = (num_tar_1_con_pie_not_sli + num_tar_1) / (num_tar_con_pie_not_sli + num_tar)
        denominator = math.sqrt(pro * (1 - pro) * (1 / num_tar_con_pie_not_sli + 1 / num_tar))

        # If denominator is zero
        if denominator == 0:
            # Update conditioned_Dic
            conditioned_Dic[index].append([list(pie_L), vote_F])
            continue

        # Get z value
        z_val = numerator / denominator
        # Get p value
        p_val = stats.norm.sf(z_val)

        # Output log file
        spamwriter_log.writerow(["check_suf_con z_val: ", z_val])
        spamwriter_log.writerow(["check_suf_con p_val: ", p_val])
        spamwriter_log.writerow('')
        f_log.flush()

        # If the pie \ slice does not significantly increase the occurrence of the target
        if p_val >= p_val_cutoff_suf:
            # Update the vote of the slice
            vote_F = -1

            # Update the sum of vote of the slices
            vote_sum += vote_F

            # Update conditioned_Dic
            conditioned_Dic[index].append([list(pie_L), vote_F])

            # Update tar_con_pie_time_LL
            tar_con_pie_time_LL = get_tar_con_pie_sli_time_LL(target, pie_L, tar_con_pie_time_LL, index)

            # Add index to the pie
            pie_L.append(index)

            return [pie_L, tar_con_pie_time_LL, eno_sam_F, suf_F]
        else:
            # Update the vote of the slice
            vote_F = 1

            # Update the sum of vote of the slices
            vote_sum += vote_F

            # Update conditioned_Dic
            conditioned_Dic[index].append([list(pie_L), vote_F])

    # Output log file
    spamwriter_log.writerow(["check_suf_con vote_sum: ", vote_sum])
    spamwriter_log.writerow('')
    f_log.flush()

    # Update suf_F
    # If more than half of the slices vote for the pie being sufficient
    if vote_sum > 0:
        suf_F = True
    else:
        suf_F = False

    return [pie_L, tar_con_pie_time_LL, eno_sam_F, suf_F]


# Get pie_vote_F_L
def get_pie_vote_F_L(pie_L, index):
    if not index in conditioned_Dic:
        return None
    else:
        pie_vote_F_LL = conditioned_Dic[index]
        for pie_vote_F_L in pie_vote_F_LL:
            if pie_equal(pie_L, pie_vote_F_L[0]):
                return pie_vote_F_L

    return None

# Check whether pie_LL is in pie_LL
def is_in(pie_L, pie_LL):
    for i in range(len(pie_LL)):
        # If the two pies are the same
        if pie_equal(pie_L, pie_LL[i]) is True:
            return True

    return False


# Check whether the two pies are the same
def pie_equal(pie_i_L, pie_j_L):
    for index in pie_i_L:
        if not index in pie_j_L:
            return False

    for index in pie_j_L:
        if not index in pie_i_L:
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


# Get P(target | pie), #(tar | pie), and #(tar = 1 | pie)
def get_pro_num_tar_con_pie(target, time_LL):
    # Initialization
    pro_tar_con_pie = None
    num_tar_con_pie = 0
    num_tar_1_con_pie = 0

    # If time_LL is None or empty
    if time_LL is None or len(time_LL) == 0:
        return [pro_tar_con_pie, num_tar_con_pie, num_tar_1_con_pie]

    # Get pro_tar_con_pie, num_tar_con_pie, and num_tar_1_con_pie
    denominator = 0

    # For each time_L
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
            # Update num_tar_con_pie, num_tar_1_con_pie, and denominator
            num_tar_con_pie += 1
            num_tar_1_con_pie += max(temp_L)
            denominator += math.pow(not_pro_Dic[target], len(temp_L))

    if denominator != 0:
        numerator = num_tar_con_pie - num_tar_1_con_pie
        pro_tar_con_pie = 1 - numerator / denominator

    return [pro_tar_con_pie, num_tar_con_pie, num_tar_1_con_pie]


# Get the value of target in the time slots
def get_tar_val_L_time_slot(target, time_LL):
    # Initialization
    tar_val_L = []

    if time_LL is None or len(time_LL) == 0:
        return tar_val_L

    # For each time_L, get the maximum value
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
def expand(target, pie_L, tar_con_pie_time_LL, root):
    # This is the slice that yields the minimum probability of the target
    min_slice = None
    # This is the minimum probability
    min_pro = None

    # X axis, Y axis and labels of X axis
    X = []
    Y = []
    X_lab = []

    # For each slice in slice_LL
    for index in range(len(slice_LL)):
        # If the slice has not been included,
        # or removed,
        # or replaced yet
        if (not index in pie_L
            and not index in removed_Dic
            and not index in replaced_Dic):

            spamwriter_log.writerow(["expand slice_LL[index]: ", slice_LL[index]])
            f_log.flush()

            # Get the list of target's value that can be changed by the slice
            tar_con_sli_time_LL = get_tar_con_pie_time_LL(target, [index])

            # Get the list of list of timepoints where the target can be changed by the pie but not the slice
            tar_con_pie_not_sli_time_LL = get_tar_con_pie_not_sli_time_LL(tar_con_pie_time_LL, tar_con_sli_time_LL)

            # Get P(target | pie \ slice)
            pro_tar_con_pie_not_sli, num_tar_con_pie_not_sli, num_tar_1_con_pie_not_sli = get_pro_num_tar_con_pie(
                target, tar_con_pie_not_sli_time_LL)
            spamwriter_log.writerow(["expand pro_tar_con_pie_not_sli: ", pro_tar_con_pie_not_sli])
            f_log.flush()

            # If P(target | pie) is None or not enough sample
            if pro_tar_con_pie_not_sli is None or num_tar_con_pie_not_sli <= sample_size_cutoff:
                continue

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
    f_log.flush()

    # Output the pie
    print(['expand pie_L: ', decode(pie_L)])

    # Update root, the first slice added to the pie
    if root is None:
        root = min_slice
        # Write root to spamwriter_log
        spamwriter_log.writerow(['root' + ': ', slice_LL[root]])
        f_log.flush()

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
    f_log.flush()

    # Backup pie_L
    backup_pie_L = [] + pie_L

    # Check each slice
    for index in backup_pie_L:
        spamwriter_log.writerow(["check_nec_con slice_LL[index]: ", slice_LL[index]])
        f_log.flush()

        # Get pie \ slice
        temp_L = [] + pie_L
        temp_L.remove(index)

        # Get the list of target's value that can be changed by pie \ slice
        tar_con_pie_time_LL = get_tar_con_pie_time_LL(target, temp_L)

        # Get the list of target's value that can be changed by the slice
        tar_con_sli_time_LL = get_tar_con_pie_time_LL(target, [index])

        # Get the list of list of timepoints where the target can be changed by the pie but not the slice
        tar_con_pie_not_sli_time_LL = get_tar_con_pie_not_sli_time_LL(tar_con_pie_time_LL, tar_con_sli_time_LL)

        # Get P(target | pie \ slice)
        pro_tar_con_pie_not_sli, num_tar_con_pie_not_sli, num_tar_1_con_pie_not_sli = get_pro_num_tar_con_pie(
            target, tar_con_pie_not_sli_time_LL)
        spamwriter_log.writerow(["check_nec_con pro_tar_con_pie_not_sli: ", pro_tar_con_pie_not_sli])
        spamwriter_log.writerow(["check_nec_con num_tar_con_pie_not_sli: ", num_tar_con_pie_not_sli])
        spamwriter_log.writerow(["check_nec_con num_tar_1_con_pie_not_sli: ", num_tar_1_con_pie_not_sli])
        f_log.flush()

        if pro_tar_con_pie_not_sli is None:
            continue

        # Get numerator
        pro_tar = pro_Dic[target]
        numerator = pro_tar_con_pie_not_sli - pro_tar
        spamwriter_log.writerow(["check_nec_con numerator: ", numerator])
        f_log.flush()

        # Get denominator
        num_tar = num_Dic[target]
        num_tar_1 = num_1_Dic[target]
        pro = (num_tar_1_con_pie_not_sli + num_tar_1) / (num_tar_con_pie_not_sli + num_tar)
        denominator = math.sqrt(pro * (1 - pro) * (1 / num_tar_con_pie_not_sli + 1 / num_tar))

        # If denominator is zero
        if denominator == 0:
            continue

        # Get z value
        z_val = numerator / denominator
        # Get p value
        p_val = stats.norm.sf(z_val)

        # Output log file
        spamwriter_log.writerow(["check_nec_con z_val: ", z_val])
        spamwriter_log.writerow(["check_nec_con p_val: ", p_val])
        spamwriter_log.writerow('')
        f_log.flush()

        # If P(target | pie \ slice) < P(target)
        if p_val < p_val_cutoff_nec:
            # Remove the slice (since it is not necessary)
            pie_L.remove(index)

            # Update replaced_Dic
            replaced_Dic[index] = 1

    return [pie_L, len(pie_L) != len(backup_pie_L)]


# Remove the influence of the pie from the data
def remove_inf(target, tar_con_pie_time_LL):
    # Remove the influence of the pie from the data
    for tar_con_pie_time_L in tar_con_pie_time_LL:
        for time in tar_con_pie_time_L:
            # If the target was changed by the pie at the current time
            if time in val_Dic[target] and val_Dic[target][time] == 1:
                val_Dic[target][time] = -1


# Shrink the pie by removing the slice that yields the maximum probability of the target that can be changed by the remaining pie but not the slice
def shrink(target, pie_L):
    # This is the slice that yields the maximum probability of the target
    max_slice = None
    # This is the maximum probability
    max_pro = None
    # This is the list of list of timepoints where the target can be changed by the remaining pie but not the slice
    max_tar_con_pie_time_LL = []

    # If the pie is None or empty
    if pie_L is None or len(pie_L) == 0:
        return [pie_L, max_tar_con_pie_time_LL, max_slice]

    # X axis, Y axis and labels of X axis
    X = []
    Y = []
    X_lab = []

    # For each slice in the pie
    for index in pie_L:
        spamwriter_log.writerow(["shrink slice_LL[index]: ", slice_LL[index]])
        f_log.flush()

        # Get pie \ slice
        temp_L = [] + pie_L
        temp_L.remove(index)

        # Get the list of list of timepoints where the target can be changed by temp_L (i.e., pie \ slice)
        tar_con_pie_time_LL = get_tar_con_pie_time_LL(target, temp_L)

        # Get the list of target's value that can be changed by the slice
        tar_con_sli_time_LL = get_tar_con_pie_time_LL(target, [index])

        # Get the list of list of timepoints where the target can be changed by the pie but not the slice
        tar_con_pie_not_sli_time_LL = get_tar_con_pie_not_sli_time_LL(tar_con_pie_time_LL, tar_con_sli_time_LL)

        # Get P(target | pie \ slice)
        pro_tar_con_pie_not_sli, num_tar_con_pie_not_sli, num_tar_1_con_pie_not_sli = get_pro_num_tar_con_pie(
            target, tar_con_pie_not_sli_time_LL)
        spamwriter_log.writerow(["shrink pro_tar_con_pie_not_sli: ", pro_tar_con_pie_not_sli])
        spamwriter_log.writerow(["shrink num_tar_con_pie_not_sli: ", num_tar_con_pie_not_sli])
        spamwriter_log.writerow(["shrink num_tar_1_con_pie_not_sli: ", num_tar_1_con_pie_not_sli])
        f_log.flush()

        # If P(target | pie \ slice) is None
        if pro_tar_con_pie_not_sli is None:
            continue

        # Update max_slice and max_pro
        if max_pro is None or max_pro < pro_tar_con_pie_not_sli:
            max_slice = index
            max_pro = pro_tar_con_pie_not_sli
            max_tar_con_pie_time_LL = tar_con_pie_time_LL

        # Update X axis, Y axis and labels of X axis
        X.append(index)
        Y.append(max_pro)
        X_lab.append(slice_LL[index][0][4:])

    # If the pie cannot be shrinked anymore
    if max_slice is None:
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

    # Remove max_slice from the pie
    pie_L.remove(max_slice)
    # Write pie_L to spamwriter_log
    spamwriter_log.writerow(['shrink pie_L' + ': ', decode(pie_L)])
    f_log.flush()

    # Output the pie
    print(['shrink pie_L: ', decode(pie_L)])

    # Update replaced_Dic
    replaced_Dic[max_slice] = 1

    return [pie_L, max_tar_con_pie_time_LL, max_slice]


# Main function
if __name__ == "__main__":
    # get parameters from command line
    # please see details of the parameters in the readme file
    src_data_file = sys.argv[1]
    tar_data_file = sys.argv[2]
    pie_file = sys.argv[3]
    log_file = sys.argv[4]
    fig_dir = sys.argv[5]
    pie_size_cutoff = int(sys.argv[6])
    p_val_cutoff_suf = float(sys.argv[7])
    p_val_cutoff_nec = float(sys.argv[8])
    sample_size_cutoff = int(sys.argv[9])
    lag_L = sys.argv[10:]

    # Initialization
    initialization(src_data_file, tar_data_file)

    with open(log_file, 'w') as f_log:
        # Write the log file
        spamwriter_log = csv.writer(f_log, delimiter=' ')
        with open(pie_file, 'w') as f_pie:
            # Write the causal pie file
            spamwriter_pie = csv.writer(f_pie, delimiter=' ')
            # Search for the causal pies
            search()
