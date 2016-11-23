

# Please cite the following paper when using the code


# Modules
from __future__ import division
from scipy import stats
# from queuelib import FifoDiskQueue
import Queue
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

# The list of sources
src_L = []

# The list of targets
trg_L = []

# The list of timepoints
time_series_LL = []

# The dictionary of data types
# key: var
# val: "discrete" or "continuous_valued"
var_type_Dic = {}

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


# Load data
# @param        src_file           source data file, which includes variables that can be the causes, the data are of the following form
#                                  var1 (i.e. header), var1_t1 (i.e. val), ..., var1_tn (i.e. val)
#                                  , ...,
#                                  varn (i.e. header), varn_t1 (i.e. val), ..., varn_tn (i.e. val)
# @param        trg_file           target data file, which includes variables that can be the effects, the data are of the following form
#                                  var1 (i.e. header), var1_t1 (i.e. val), ..., var1_tn (i.e. val)
#                                  , ...,
#                                  varn (i.e. header), varn_t1 (i.e. val), ..., varn_tn (i.e. val)
# @param        transpose_F          tells us whether the data need to be transposed
#                                  False, when the data are of the above form
#                                  True,  when the data are of the following form
#                                  var1 (i.e. header), ..., varn (i.e. header)
#                                  var1_t1 (i.e. val), ..., varn_t1 (i.e. val)
#                                  , ...,
#                                  var1_tn (i.e. val), ..., varn_tn (i.e. val)
def initialization(src_file, trg_file, transpose_F):
    # Load source file
    load_data(src_file, transpose_F, True)

    # Load target file
    load_data(trg_file, transpose_F, False)


# get [var, [time, val]]
# @param        data_file          source / target file
#                                  the data are of the following form
#                                  var1 (i.e. header), var1_t1 (i.e. val), ..., var1_tn (i.e. val)
#                                  , ...,
#                                  varn (i.e. header), varn_t1 (i.e. val), ..., varn_tn (i.e. val)
# @param        transpose_F          tells us whether the data need to be transposed
#                                  False, when the data are of the above form, thus do not need to be transposed
#                                  True,  when the data are of the following form, thus need to be transposed
#                                  var1 (i.e. header), ..., varn (i.e. header)
#                                  var1_t1 (i.e. val), ..., varn_t1 (i.e. val)
#                                  , ...,
#                                  var1_tn (i.e. val), ..., varn_tn (i.e. val)
# @param        src_F              Flag variable              
#                                  True,  if target data
#                                  False, if source data
def load_data(data_file, transpose_F, src_F):
    with open(data_file, 'rb') as f:
        spamreader = list(csv.reader(f, delimiter = ','))
        if transpose_F:
            # Transpose the data
            spamreader = zip(*spamreader)

        # Get time_series_LL
        # That is, the time in [0, max_time)
        max_time = 0
        for i in range(len(spamreader)):
            if max_time < len(spamreader[i]):
                max_time = len(spamreader[i])
        for time in range(max_time):
            time_series_LL.append(time)

        # From the first line to the last
        for i in range(len(spamreader)):
            # The var's name lies in the first column in each row
            var = spamreader[i][0].strip()

            # Get the type of var (discrete or continuous_valued)
            # Default is discrete
            cont_F = False
            for time in range(1, len(spamreader[i])):
                if spamreader[i][time]:
                    val = spamreader[i][time].strip()
                    # If continuous_valued
                    if is_number(val):
                        # var is continuous_valued, flip cont_F 
                        cont_F = True
                    break

            # If continuous_valued data, update src_L (when src_F = True, i.e. source file) or trg_L (when src_F = False, i.e. trg file), update var_type_Dic
            if cont_F:
                if src_F:
                    src_L.append(var)
                else:
                    trg_L.append(var)
                var_type_Dic[var] = "continuous_valued"

            # Get val_Dic, update src_L (when src_F = True, i.e. source file) or trg_L (when src_F = False, i.e. trg file), update var_type_Dic
            for time in range(1, len(spamreader[i])):
                if spamreader[i][time]:
                    val = spamreader[i][time].strip()
                    # If continuous_valued
                    if is_number(val):
                        # Update val_Dic
                        if not var in val_Dic:
                            val_Dic[var] = {}
                        val_Dic[var][time] = float(val)
                    # If discrete
                    else:
                        # Update var and val_Dic
                        var_val = var + "_" + val
                        if not var_val in val_Dic:
                            val_Dic[var_val] = {}
                        # This says var is True at the time
                        val_Dic[var_val][time] = 1
                        # Update src_L or trg_L
                        if src_F:
                            if not var_val in src_L:
                                src_L.append(var_val)
                        elif not var_val in trg_L:
                            trg_L.append(var_val)
                        # Update var_type_Dic
                        var_type_Dic[var_val] = "discrete"


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
    for src in src_L:
        for win_L in win_LL:
            slice_L = [src, win_L[0], win_L[1]]
            slice_LL.append(slice_L)


# Search for the candidates of causal pies
def search():
    for target in trg_L:
        # Used for BFS
        q = Queue.Queue()

        # Initialization, add all slices to queue
        for i in range(len(slice_LL)):
            q.put([i])

        # BFS
        while not q.empty():
            size = q.qsize()
            for i in range(size):
                # Poll a pie from queue
                pie_L = q.get()

                # Indicated whether the pie has been expanded
                # If not, it is a candidate as long as its size > 1 (i.e. not a single slice)
                # Default is False
                expanded_F = False

                # Get the index of the last slice
                index = pie_L[len(pie_L) - 1]

                # Visit slices after slice_L[index]
                for j in range(index + 1, len(slice_LL)):
                    # The new pie including the slice
                    pie_inc_L = pie_L + [j]
                    # Get sample with respect to [pie_L \wedge j] (i.e. the pie including the slice)
                    val_inc_L = get_pie_A_not_B_val_L(target, pie_inc_L, None)

                    # Get sample with respect to [pie_L \wedge \neg j] (i.e. the pie excluding the slice)
                    val_exc_S_L = get_pie_A_not_B_val_L(target, pie_L, [j])

                    # Get sample with respect to [slice_L \wedge \neg pie_L] (i.e. the slice excluding the pie)
                    val_exc_P_L = get_pie_A_not_B_val_L(target, [j], pie_L)

                    # Interviewer / Interviewee test
                    if II_test(val_inc_L, val_exc_S_L) and II_test(val_inc_L, val_exc_P_L):
                        # Add the new pie to the queue
                        q.add(P_Inc_L)
                        expand_F = True

                # If the pie cannot be expanded and its size is larger than 1
                if not expand_F and len(pie_L) > 1:
                    if target not in pie_Dic:
                        pie_Dic[target] = []
                    pie_Dic[target].append(pie_L)


def get_pie_A_not_B_val_L(target, pie_A_L, pie_B_L):
    pie_A_time_LL = get_pie_time_LL(pie_A_L)
    pie_B_time_LL = get_pie_time_LL(pie_B_L)
    pie_A_not_B_time_LL = get_pie_A_not_B_time_LL(pie_A_time_LL, pie_B_time_LL)
    print([pie_A_L, pie_A_time_LL])
    print([pie_B_L, pie_B_time_LL])
    print(pie_A_not_B_time_LL)
    val_L = get_val_L(target, pie_A_not_B_time_LL)
    return val_L


def get_pie_time_LL(pie_L):
    print(["pie_L", pie_L])
    print(time_series_LL)
    # If pie_L is None, return pie_time_LL
    if not pie_L:
        return [[]]

    # Initialization
    pie_time_LL = []
    # Get dictionary of start and end
    [start_Dic, end_Dic] = get_start_end_Dic(pie_L)
    print(["start_Dic", start_Dic])
    print(["end_Dic", end_Dic])
    # Get pie_time_Dic
    # Key: var
    # Value: number of times the var occurs
    pie_time_Dic = {}
    # Flag, indicating whether we have started recording the timepoints where all the slices in the pie are present
    # Default is None
    recorded_F = False
    # Get pie_time_LL 
    for time in time_series_LL:
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
        print(pie_time_Dic)
        print[time, len(pie_time_Dic)]
        # If all the slices in the pie are present
        if len(pie_time_Dic) == len(pie_L):
            if not recorded_F:
                recorded_F = True
                time_L = []
            time_L.append(time)
        # If some slices are absent and we have been recording time
        elif recorded_F:
            pie_time_LL.append(time_L)
            recorded_F = False
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
            start_time = time + win_start
            end_time = time + win_end
            start_Dic[index][start_time] = 1
            end_Dic[index][end_time] = 1
    return [start_Dic, end_Dic]


# Get the time when pie A is present whereas pie B is absent
def get_pie_A_not_B_time_LL(pie_A_time_LL, pie_B_time_LL):
    # Get time_Dic
    pie_B_time_Dic = {}
    for pie_B_time_L in pie_B_time_LL:
        for time in pie_B_time_L:
            pie_B_time_Dic[time] = 1

    # Get pie_A_not_B_time_LL
    recorded_F = False
    pie_A_not_B_time_LL = []
    for pie_A_time_L in pie_A_time_LL:
        pie_A_not_B_time_L = []
        for time in pie_A_time_L:
            if not time in pie_B_time_Dic:
                if not recorded_F:
                    pie_A_not_B_time_L = []
                pie_A_not_B_time_L.append(time)
            elif recorded_F:
                pie_A_not_B_time_LL.append(pie_A_not_B_time_L)


# Get the value of target in the time slots
def get_val_L(target, time_LL):
    print(target)
    print(time_LL)
    # Initialization
    val_L = []
    # For each time_L, get the maximum absolute value
    for time_L in time_LL:
        # Initialization
        max_abs_val = None
        for time in time_L:
            if time in val_Dic[target] and (not max_abs_val or abs(max_abs_val) < abs(val_Dic[target][time])):
                max_abs_val = val_Dic[target][time]
        if var_type_Dic[target] == "discrete":
            if not max_abs_val:
                val_L.append(0)
            else:
                val_L.append[max_abs_val]
        elif max_abs_val:
                val_L.append[max_abs_val]


# Interviewer / Interviewee test
def II_test(target, val_inc_L, val_exc_L):
    # Unpaired t test
    t, p = stats.ttest_ind(val_inc_L, val_exc_L, equal_var = False)

    # If target is discrete
    if target in disc_Dic:
        # If compared to exclusion, inclusion does not significantly increase e's occurrence
        if t <= 0 or p >= threshold:
            # Will not include
            return False
    # If 1) target is continuous_valued and 2) inclusion does not significantly increase or decrease e's value
    elif p >= threshold:
            # Will not include
            return False
    # Include
    return True


# Output candidates
def output_candidates():
    with open(pie_file, 'wb') as f:
        spamwriter = csv.writer(f, delimiter = ',')
        for trg in pie_Dic:
            spamwriter.writerow(pie_Dic[trg])


# Main function
if __name__=="__main__":
    # get parameters from command line
    # please see details of the parameters in the readme file
    src_file = sys.argv[1]
    trg_file = sys.argv[2]
    transpose_F = sys.argv[3]
    pie_file = sys.argv[4]
    sig_pie_file = sys.argv[5]
    pie_size_cutoff = int(sys.argv[6])
    p_val_cutoff = float(sys.argv[7])
    sample_size_cutoff = int(sys.argv[8])
    lag_L = sys.argv[9:]

    # Make directory
    directory = os.path.dirname(pie_file)
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = os.path.dirname(sig_pie_file)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Initialization
    initialization(src_file, trg_file, transpose_F)

    # Get windows
    get_win_LL(lag_L)

    # Get slices
    get_slice_LL()

    # Get candidates
    search()

    # Output candidates
    output_candidates()

    # Output significant candidates
    # output_sig_candidates()
