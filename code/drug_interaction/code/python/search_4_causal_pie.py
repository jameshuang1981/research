
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
found_Dic = {}

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

        # Iterative Deepening Search
        ids(target)


# Iterative Deepening Search
def ids(target):
    for pie_size_cutoff in range(len(slice_LL), len(slice_LL) + 1):
        # The dictionary records the slices in a found causal pie
        # key: slice
        # val: 1
        global found_Dic
        found_Dic = {}

        # Flag, indicating whether pie_size_cutoff has been met, False by default
        pie_size_cutoff_met_F = False

        while len(found_Dic) < len(slice_LL):
            # Depth Limited Search
            pie_size_cutoff_met_tem_F = dls(target, pie_size_cutoff)

            # If pie_size_cutoff has been met
            if pie_size_cutoff_met_tem_F is True:
                pie_size_cutoff_met_F = True

        if pie_size_cutoff_met_F is False:
            break


# Depth Limited Search
def dls(target, pie_size_cutoff):
    # Flag, indicating whether pie_size_cutoff has been met
    pie_size_cutoff_met_F = False

    # The first slice added to the pie, None by default
    root = None

    # The list of slices in the pie, empty by default
    pie_L = []

    # The list of list of timepoints where the target can be changed by the pie
    tar_con_pie_time_LL = get_tar_con_pie_time_LL(target, pie_L)

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

    while True:
        # Check the sufficient condition (to produce the target)
        # Flag sample_size_cutoff_met_F, indicating whether there is enough sample
        # Flag suf_F, indicating whether the pie is sufficient
        pie_L, tar_con_pie_time_LL, sample_size_cutoff_met_F, suf_F = check_suf_con(target, pie_L, tar_con_pie_time_LL)

        # The loop continues if
        #     1) the size of the pie is smaller than pie_size_cutoff,
        # and 2) there is enough sample,
        # and 3) the pie is not sufficient
        while (get_num_of_uni_nam(pie_L) < pie_size_cutoff
               and sample_size_cutoff_met_F is False
               and suf_F is False):
            # The size of the pie before expanding
            pie_size_bef_exp = len(pie_L)

            # Expand the pie
            [pie_L, tar_con_pie_time_LL] = expand(target, pie_L, tar_con_pie_time_LL)

            # If the pie cannot be expanded anymore
            if len(pie_L) == pie_size_bef_exp:
                break

            # Update root
            # If the root is None
            if root is None:
                root = pie_L[0]

            # Check the sufficient condition
            # Flag sample_size_cutoff_met_F, indicating whether there is enough sample
            # Flag suf_F, indicating whether the pie is sufficient
            pie_L, tar_con_pie_time_LL, sample_size_cutoff_met_F, suf_F = check_suf_con(target, pie_L, tar_con_pie_time_LL)

        # If pie_size_cutoff has been met, update pie_size_cutoff_met_F
        if get_num_of_uni_nam(pie_L) >= pie_size_cutoff:
            pie_size_cutoff_met_F = True

        # If enough sample and the pie is sufficient
        if sample_size_cutoff_met_F is False and suf_F is True:
            # The size of the pie before checking the necessary condition (to produce the target)
            pie_size_bef_check_nec_con = len(pie_L)

            # Check the necessary condition and remove unnecessary slices
            pie_L = check_nec_con(target, pie_L)

            # Initialization
            if not target in pie_Dic:
                pie_Dic[target] = []

            # Add the pie to pie_Dic
            pie_Dic[target].append(pie_L)

            # Update found_Dic
            # Mark each slice in the pie and the duplicated ones as removed (i.e, adding the key to the dict)
            for index in range(len(slice_LL)):
                if duplicate(pie_L, index):
                    found_Dic[index] = 1

            # Remove the influence of the pie from the data
            remove_inf(target, tar_con_pie_time_LL)

            # Write the pie to spamwriter_pie
            spamwriter_pie.writerow(['causal pie of ' + target + ': ', decode(pie_L)])
            f_pie.flush()

            # Output the pie
            print(['causal pie of ' + target + ': ', decode(pie_L)])

            break
        else:
            # The size of the pie before shrinking
            pie_size_bef_shr = len(pie_L)

            # Shrink the pie
            [pie_L, tar_con_pie_time_LL] = shrink(target, pie_L)

            # If the pie cannot be shrinked anymore
            if len(pie_L) == pie_size_bef_shr:
                # Mark the root as found (i.e, adding the key to the dict)
                found_Dic[root] = 1

                break

    return pie_size_cutoff_met_F


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

    # Flag, indicating whether there is enough sample, False by default
    sample_size_cutoff_met_F = False
    # Flag, indicating whether the pie is sufficient, False by default
    suf_F = False

    # If the pie is None or empty
    if pie_L is None or len(pie_L) == 0:
        return [pie_L, tar_con_pie_time_LL, sample_size_cutoff_met_F, suf_F]

    # Get P(target | pie)
    pro_tar_con_pie, num_tar_con_pie, num_tar_1_con_pie = get_pro_num_tar_con_pie(target, tar_con_pie_time_LL)
    spamwriter_log.writerow(["check_suf_con pro_tar_con_pie: ", pro_tar_con_pie])
    spamwriter_log.writerow(["check_suf_con num_tar_con_pie: ", num_tar_con_pie])
    spamwriter_log.writerow(["check_suf_con num_tar_1_con_pie: ", num_tar_1_con_pie])
    f_log.flush()

    # If P(target | pie) is None or not enough sample
    if pro_tar_con_pie is None or num_tar_con_pie <= sample_size_cutoff:
        # Update sample_size_cutoff_met_F, since there is no enough sample
        sample_size_cutoff_met_F = True
        return [pie_L, tar_con_pie_time_LL, sample_size_cutoff_met_F, suf_F]

    # If P(target | pie) < 0
    if pro_tar_con_pie < 0:
        return [pie_L, tar_con_pie_time_LL, sample_size_cutoff_met_F, suf_F]

    # Behrens-Welch test (unpaired t test)
    mean1 = pro_tar_con_pie
    std1 = math.sqrt(mean1 * (1 - mean1))
    nobs1 = num_tar_con_pie

    mean2 = pro_Dic[target]
    std2 = math.sqrt(mean2 * (1 - mean2))
    nobs2 = num_Dic[target]

    t_val, p_val = stats.ttest_ind_from_stats(mean1, std1, nobs1, mean2, std2, nobs2)

    # Output log file
    spamwriter_log.writerow(["check_suf_con p_val: ", t_val])
    spamwriter_log.writerow(["check_suf_con p_val: ", p_val])
    spamwriter_log.writerow('')
    f_log.flush()

    if (t_val is None or p_val is None
        # If the pie does not significantly increase the occurrence of the target
        or t_val < 0 or p_val / 2 >= p_val_cutoff_suf):
        return [pie_L, tar_con_pie_time_LL, sample_size_cutoff_met_F, suf_F]

    # Check the sufficiency conditioned on pie \ slice
    for index in range(len(slice_LL)):
        # If the slice is duplicate (i.e., has the same name with some slice in the pie)
        if duplicate(pie_L, index):
            continue

        # Get pie_vote_F_L
        pie_vote_F_L = get_pie_vote_F_L(pie_L, index)

        # If the slice has been used
        if pie_vote_F_L is not None:
            # Get the vote of the slice
            vote_F =  pie_vote_F_L[1]
            if vote_F == -1:
                return [pie_L, tar_con_pie_time_LL, sample_size_cutoff_met_F, suf_F]

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

        # If P(target | pie) is None
        if pro_tar_con_pie_not_sli is None:
            # The slice cannot vote for insufficiency
            vote_F = 0

            # Update conditioned_Dic
            conditioned_Dic[index].append([list(pie_L), vote_F])

            continue

        # If P(target | pie \ slice) < 0
        if pro_tar_con_pie_not_sli < 0:
            # The slice can vote for insufficiency
            vote_F = -1

            # Update conditioned_Dic
            conditioned_Dic[index].append([list(pie_L), vote_F])

            return [pie_L, tar_con_pie_time_LL, sample_size_cutoff_met_F, suf_F]

        # Behrens-Welch test (unpaired t test)
        mean1 = pro_tar_con_pie_not_sli
        std1 = math.sqrt(mean1 * (1 - mean1))
        nobs1 = num_tar_con_pie_not_sli

        t_val, p_val = stats.ttest_ind_from_stats(mean1, std1, nobs1, mean2, std2, nobs2)

        # Output log file
        spamwriter_log.writerow(["check_suf_con z_val: ", t_val])
        spamwriter_log.writerow(["check_suf_con p_val: ", p_val])
        spamwriter_log.writerow('')
        f_log.flush()

        if (t_val is None or p_val is None
            # If the pie \ slice still significantly increases the occurrence of the target
            or t_val > 0 and p_val / 2 < p_val_cutoff_suf):
            # The slice cannot vote for insufficiency
            vote_F = 0

            # Update conditioned_Dic
            conditioned_Dic[index].append([list(pie_L), vote_F])
        # If the pie \ slice does not significantly increase the occurrence of the target
        else:
            # Update the vote of the slice, which votes that the pie is not sufficient
            vote_F = -1

            # Update conditioned_Dic
            conditioned_Dic[index].append([list(pie_L), vote_F])

            return [pie_L, tar_con_pie_time_LL, sample_size_cutoff_met_F, suf_F]

    # Update suf_F
    suf_F = True

    return [pie_L, tar_con_pie_time_LL, sample_size_cutoff_met_F, suf_F]


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


# Check whether index is a slice
def is_a_sli(index, pie_LL):
    for i in range(len(pie_LL)):
        # If the slice is in the pie
        if index in pie_LL[i]:
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


# Check whether the slice is duplicate
def duplicate(pie_L, index):
    # For each slice in the pie
    for index_pie in pie_L:
        # If the name of the two variables are the same
        if slice_LL[index_pie][0] == slice_LL[index][0]:
            return True

    return False


# Get the number of unique names of the slices in the pie
def get_num_of_uni_nam(pie_L):
    # The dictionary that keeps track of the unique names
    uni_nam_Dic = {}

    for index in pie_L:
        var, win_start, win_end = slice_LL[index]
        if not var in uni_nam_Dic:
            uni_nam_Dic[var] = 1

    return len(uni_nam_Dic)


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
def expand(target, pie_L, tar_con_pie_time_LL):
    # This is the slice that yields the minimum probability of the target
    min_slice = None
    # This is the minimum probability
    min_pro = None

    # For each slice in slice_LL
    for index in range(len(slice_LL)):
        # If the slice has not been included,
        # or removed,
        # or replaced yet
        if (not index in pie_L
            and not index in found_Dic
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
            spamwriter_log.writerow(["expand num_tar_con_pie_not_sli: ", num_tar_con_pie_not_sli])
            spamwriter_log.writerow(["expand num_tar_1_con_pie_not_sli: ", num_tar_1_con_pie_not_sli])
            f_log.flush()

            # If P(target | pie) is None or not enough sample
            if pro_tar_con_pie_not_sli is None or num_tar_con_pie_not_sli <= sample_size_cutoff:
                continue

            # Update min_slice and min_pro
            if min_pro is None or min_pro > pro_tar_con_pie_not_sli:
                min_slice = index
                min_pro = pro_tar_con_pie_not_sli

    # If the pie cannot be expanded anymore
    if min_slice is None:
        return [pie_L, tar_con_pie_time_LL]

    # Update tar_con_pie_time_LL
    tar_con_pie_time_LL = get_tar_con_pie_sli_time_LL(target, pie_L, tar_con_pie_time_LL, min_slice)

    # Add min_slice to the pie
    pie_L.append(min_slice)
    # Write pie_L to spamwriter_log
    spamwriter_log.writerow(['expand pie_L' + ': ', decode(pie_L)])
    f_log.flush()

    # Output the pie
    print(['expand pie_L: ', decode(pie_L)])

    return [pie_L, tar_con_pie_time_LL]


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
    backup_pie_L = list(pie_L)

    # Check each slice
    for index in backup_pie_L:
        spamwriter_log.writerow(["check_nec_con slice_LL[index]: ", slice_LL[index]])
        f_log.flush()

        # Get pie \ slice
        temp_L = list(pie_L)
        temp_L.remove(index)

        # Get the list of target's value that can be changed by pie \ slice
        tar_con_pie_time_LL = get_tar_con_pie_time_LL(target, temp_L)

        # Check the sufficient condition (to produce the target)
        # Flag sample_size_cutoff_met_F, indicating whether there is enough sample
        # Flag suf_F, indicating whether the pie is sufficient
        temp_L, tar_con_pie_time_LL, sample_size_cutoff_met_F, suf_F = check_suf_con(target, temp_L, tar_con_pie_time_LL)

        # If the pie \ slice still significantly increases the occurrence of the target
        if suf_F is True:
            # Remove the slice (since it is not necessary)
            pie_L.remove(index)

            # Update replaced_Dic
            replaced_Dic[index] = 1

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
def shrink(target, pie_L):
    # This is the slice that yields the maximum probability of the target
    max_slice = None
    # This is the maximum probability
    max_pro = None
    # This is the list of list of timepoints where the target can be changed by the remaining pie but not the slice
    max_tar_con_pie_time_LL = []

    # If the pie is None or empty
    if pie_L is None or len(pie_L) == 0:
        return [pie_L, max_tar_con_pie_time_LL]

    # For each slice in the pie
    for index in pie_L:
        spamwriter_log.writerow(["shrink slice_LL[index]: ", slice_LL[index]])
        f_log.flush()

        # Get pie \ slice
        temp_L = list(pie_L)
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

        # If P(target | pie) is None
        if pro_tar_con_pie_not_sli is None:
            continue

        # Update max_slice and max_pro
        if max_pro is None or max_pro < pro_tar_con_pie_not_sli:
            max_slice = index
            max_pro = pro_tar_con_pie_not_sli
            max_tar_con_pie_time_LL = tar_con_pie_time_LL

    # If the pie cannot be shrinked anymore
    if max_slice is None:
        return [pie_L, max_tar_con_pie_time_LL]

    # Remove max_slice from the pie
    pie_L.remove(max_slice)
    # Write pie_L to spamwriter_log
    spamwriter_log.writerow(['shrink pie_L' + ': ', decode(pie_L)])
    f_log.flush()

    # Output the pie
    print(['shrink pie_L: ', decode(pie_L)])

    # Update replaced_Dic
    replaced_Dic[max_slice] = 1

    return [pie_L, max_tar_con_pie_time_LL]


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
    sample_size_cutoff = int(sys.argv[8])
    lag_L = sys.argv[9:]

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