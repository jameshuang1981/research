from __future__ import division
import sys
import os
import csv
import numpy as np

# This is the code for calculating causal significance using alpha

# Define global variables

# key: [c r s], i.e. a list including potential cause c, start and end of time window, r and s  
# val: [T(e | c_t)], i.e. a list of T(e | c_t), which is the set of timepoints where e is measured in c_t's window (c_t is the instance of c occurring at time t)
T_e_ct_LL_Dic = {}

# key: [c r s], i.e. a list including potential cause c, start and end of time window, r and s
# val: T(e | c), which is the set of timepoints where e is measured in c's time window
T_e_c_L_Dic = {}

# key: [c r s], i.e. a list including potential cause c, start and end of time window, r and s
# val: T(c), which is the set of timepoints where c occurs 
T_c_L_Dic = {}

# key: [c r s], i.e. a list including potential cause c, start and end of time window, r and s
# val: N(e | c, x), the total number of e being measured in the intersection of c and x's windows
N_e_c_x_Dic = {}

# key: [c r s], i.e. a list including potential cause c, start and end of time window, r and s
# val: N(e | c), the total number of e being measured in c's window
N_e_c_Dic = {}

# key: [c r s], i.e. a list including potential cause c, start and end of time window, r and s
# val: E[e | c], the expectation of e conditioned on c
E_e_c_Dic = {}

# key: e, an effect
# val: E[e], the expectation of e
E_e_Dic = {}

# key: [c r s], i.e. a list including potential cause c, start and end of time window, r and s
# val: true or false
# true, if c is the potential cause of e measured at time t
# false, otherwise
is_in_Xt_L_Dic = {}

# key: t, the time of e being measured
# val: X_t, the set of potential causes of e measured at time t 
Xt_L_Dic = {}

# key: time t
# val: list of discrete or discretized vars occurring at t
disc_val_Dic = {}

# key: continuous_valued var
# val: list of var's continuous value at a time
cont_val_L_Dic = {}

# key: continuous_valued var
# val: list of time and var's continuous value at a time, i.e. (list (list time val))
time_cont_val_L_Dic = {}

# list of discrete or discretized vars in the time series
alphabet_disc = []

# list of continuous_valued vars in the time series
alphabet_cont = []

# key: [effect]
# val: list of [c r s] where c is a potential cause, r and s the start and end of time window
relations = {}

# Dictionaries for the greedy version of is_full_rank (i.e. is_full_rank_greedy) 
max_row_Dic = {}
coefficient_Dic = {}
denominator_Dic = {}
multiplier_Dic = {}
# calculate causal significance for all relationships
# @param        result_file         file containing relationships and their causal significance
def get_all_alpha(result_file):
  # output heading
  with open(result_file, 'wb') as f:
    spamwriter = csv.writer(f, delimiter = ',')
    spamwriter.writerow(["cause", "effect", "window_start", "window_end", "alpha"])

  for e in relations:
    get_alpha(e, result_file)


# calculate alpha for one relationship
# @param        effect_L            (list e), a list of e
# @param        result_file         file containing relationships and their causal significance
def get_alpha(e, result_file):
  # get the set of potential causes and remove duplicates
  X_L = set(relations[e])
  if X_L:
    A_array = get_A_array(X_L)
    # temp_array is the same as A_array and is used to check whether A_array is full rank
    temp_array = get_A_array(X_L)
    if is_full_rank(temp_array):
      # if A_array is full rank, solve system of linear equations
      B_array = get_B_array(X_L, e)
      B_array = solve_system_of_linear_equations(A_array, B_array)
      get_result_file(X_L, e, B_array, result_file)
    else:
      # if A_array is not full rank, get the subsystem of linear equations
      X_LIS_L = get_X_LIS_L(X_L, e)
      if X_LIS_L:
        # solve the subsystem of linear equations
        A_LIS_array = get_A_array(X_LIS_L)
        B_LIS_array = get_B_array(X_LIS_L)
        B_LIS_array = solve_system_of_linear_equations(A_LIS_array, B_LIS_array)
        get_result_file(X_LIS_L, e, B_LIS_array, result_file)


# get A_array, the coefficient matrix of the system of linear equations
# @param        X_L                 the set of potential causes
def get_A_array(X_L):
  n = len(X_L)
  A_array = np.zeros(n * n)
  for i in range(n):
    for j in range(n):
      c_L = X_L[i]
      x_L = X_L[j]
      N_e_c_x = get_N_e_c_x(c_L, x_L)
      N_e_x = get_N_e_c(x_L)
      T_e_c_L = get_T_e_c_L(c_L)
      N_e_c = get_N_e_c(c_L)
      len_T_e = len(disc_val_Dic)
      nominator = N_e_c_x * len_T_e - N_e_x * len(T_e_c_L)
      denominator = N_e_c * (len_T_e - len(T_e_c_L))
      f_e_c_x = nominator / denominator
      A_array[i * n + j] = f_e_c_x
  return A_array


# get *N_e_c_x_Dic*
# @param        c_L                [c, r, s], a list including potential cause c, start and end of time window, r and s
# @param        x_L                (list x r s), a list including potential cause x, start and end of time window, r and s
def get_N_e_c_x(c_L, x_L):
  if not [c_L, x_L] in N_e_c_x_Dic:
    T_e_xt_LL = get_T_e_ct_LL(x_L)
    N_e_c_x = 0
    for T_e_xt_L in T_e_xt_LL:
      for time in T_e_xt_L:
        if is_in_Xt(c_L, time):
          N_e_c_x += 1
    N_e_c_x_Dic[[c_L, x_L]] = N_e_c_x
  return N_e_c_x_Dic[[c_L, x_L]]


# get *T_e_ct_LL_Dic*
# @param        c_L                [c, r, s], a list including potential cause c, start and end of time window, r and s
def get_T_e_ct_LL(c_L):
  if not c_L in T_e_ct_LL_Dic:
    c, r, s = c_L
    T_c_L = get_T_c_L(c)
    T_e_ct_LL = []
    for tc in T_c_L:
      T_e_ct_L = []
      for time in range(tc + r, tc + s + 1):
        if time in disc_val_Dic:
          T_e_ct_L.append(time)
      if T_e_ct_L:
        T_e_ct_LL.append(T_e_ct_L)
    T_e_ct_LL_Dic[c_L] = T_e_ct_LL
  return T_e_ct_LL_Dic[c_L]


# get *T_c_L_Dic*
# @param        c                   a potential cause 
def get_T_c_L(c):
  if not c in T_c_L_Dic:
    T_c_L = []
    for time in disc_val_Dic:
      if c in disc_val_Dic[time]:
        T_c_L.append(time)
    T_c_L_Dic[c] = T_c_L
  return T_c_L_Dic[c]


# get *T_e_c_L_Dic*
# @param        c_L                [c, r, s], a list including potential cause c, start and end of time window, r and s
def get_T_e_c_L(c_L):
  if not c_L in T_e_c_L_Dic:
    T_e_ct_LL = get_T_e_ct_LL(c_L)
    T_e_c_L = []
    for T_e_ct_L in T_e_ct_LL:
      for time in T_e_ct_L:
        T_e_c_L.append(time)
    T_e_c_L = set(T_e_c_L)
    T_e_c_L_Dic[c_L] = T_e_c_L
  return T_e_c_L_Dic[c_L]


# get *N_e_c_Dic*
# @param        c_L                [c, r, s], a list including potential cause c, start and end of time window, r and s
def get_N_e_c(c_L):
  if not c_L in N_e_c_Dic:
    T_e_ct_LL = get_T_e_ct_LL(c_L)
    N_e_c = 0
    for T_e_ct_L in T_e_ct_LL:
      N_e_c += len(T_e_ct_L)
    N_e_c_Dic[c_L] = N_e_c
  return N_e_c_Dic[c_L]


# is A_array full rank?
# @param        A_array             the coefficient matrix of the system of linear equations
def is_full_rank(A_array):
  n = int(math.sqrt(len(A_array)))
  full_rank = True
  # gaussian elimination
  for k in range(n - 1):
    # find the max row and switch it with row k
    max_val = abs(A_array[k * n + k])
    max_row = k
    # find the max row
    for i in range(k + 1, n):
      if abs(A_array[i * n + k]) > max_val:
        max_val = abs(A_array[ i * n + k])
        max_row = i
    if max_val = 0:
      full_rank = False
      break
    # switch it with row k
    if k != max_row:
      for j in range(k, n):
        A_array[max_row * n + j], A_array[k * n + j] = A_array[k * n + j], A_array[max_row * n + j]
    # division
    denominator = A_array[k * n + k]
    for j in range(k, n):
      A_array[k * n + j] = A_array[k * n + j] / denominator
    # subtraction
    for i in range(k + 1, n):
      multiplier = A_array[i * n + k]
      for j in range(k, n):
        A_array[i * n + j] -= A_array[k * n + j] * multiplier
  # return
  if full_rank == False or A_array[(n - 1) * n + (n - 1)] == 0:
    return False
  else:
    return True


# check whether A is full rank in a greedy fashion
# @param        A_array             the coefficient matrix of the system of linear equations
def is_full_rank_greedy(A_array):
  N = int(math.sqrt(len(A_array)))
  N_local = N - 2
  full_rank = True

  # gaussian elimination
  for k in range(N - 1):
    # update the denominator
    if N_local > 0 and k == N_local:
      A_array[N_local * N + N_local] = coefficient_Dic[[N_local, N_local]]
    # find the max row and switch it with row k
    max_row = k
    if k >= N_local:
      max_val = abs(A_array[k * N + k])
      # find the max row
      for i in range(K + 1, N):
        if abs(A_array[i * N + k]) > max_val:
          max_val = abs(A_array[i * N + k])
          max_row = i
      # check rank
      if max == 0:
        full_rank = False
        break
      # record max row
      max_row_Dic[k] = max_row
    else:
      max_row = max_row_Dic[k]

    # switch it with row k
    if k != max_row:
      for j in range(k, N):
        A_array[max_row * N + j], A_array[k * N + j] = A_array[k * N + j], A_array[max_row * N + j]

    # division
    if not k in denominator_Dic:
      denominator_Dic[k] = A_array[k * N + k]
    denominator = denominator_Dic[k]
    for j in range(k, N):
      if j > N_local or (j == k and j == N_local):
        A_array[k * N + j] /= denominator

    # subtraction
    for i in range(k + 1, N):
      if not [i, k] in multiplier_Dic:
        multiplier_Dic[[i, k]] = A_array[i * N + k]
      multiplier = multiplier_Dic[[i, k]]
      for j in range(k, N):
        if i > N_local and j > N_local:
          if j <= N_local and k < N_local:
            A_array[i * N + j] -= coefficient_Dic[[k, j]] * multiplier
          else:
            A_array[i * N + j] -= A_array[k * N + j] * multiplier

    # return
    if full_rank == False or A_array[(N - 1) * N) + N - 1] == 0:
      return False
    else:
      # update coefficient_Dic
      for i in range(N):
        for j in range(N):
          if i > N_local or j > N_local or (i == j and i == N_local):
            coefficient_Dic[[i, j]] = A_array[i * N + j]
      return True


# get B_array, the value vector of the system of linear equations
# @param        A_array             the coefficient matrix of the system of linear equations
# @param        X_L                 the set of potential causes
# @param        e                   an effect
def get_B_array(X_L e):
  n = len(X_L)
  B_array = np.zero(n)
  for i in range(n):
    c_L = X_L[i]
    T_e_c_L = ge_T_e_c_L(c_L)
    N_e_c = get_N_e_c(c_L)
    E_e_c = get_E_e_c(e, c_L)
    E_e = get_E_e(e)
    len_T_e = len(disc_val_Dic)
    nominator = len_T_e * len(T_e_c_L)
    denominator = N_e_c * (len_T_e - len(T_e_c_L))
    f_e_c = nominator / denominator
    B_array[i] = f_e_c * (E_e_c - E_e)
  return B_array


# is c in Xt? i.e. is c a potential cause of e being measured at time t
# @param        c_L                [c, r, s], a list including potential cause c, start and end of time window, r and s
# @param        time               the time where e is measured
def is_in_Xt(c_L, time):
  if not c_L in is_in_Xt_L_Dic:
    T_e_c_L = get_T_e_c_L(c_L)
    for te in T_e_c_L:
      Xt_L_Dic[te].append(c_L)
    is_in_Xt_L_Dic[c_L] = True
  if c_L in Xt_L_Dic:
    return True
  else:
    return False


# solve system of linear equations
# @param        A_array             the coefficient matrix of the system of linear equations
# @param        A_array             the value vector of the system of linear equations
def solve_system_of_linear_equations(A_array, B_array):
  n = int(math.sqrt(len(A_array)))
  # gaussian elimination
  for k in range(n - 1):
    # find the max row and switch it with row k
    max_val = abs(A_array[k * n + k])
    max_row = k
    # find the max row
    for i in range(k + 1, n):
      if abs(A_array[i * n + k]) > max_val:
        max_val = abs(A_array[i * n + k])
        max_row = i
    # switch it with row k
    if k /= max_row:
      for j in range(k, n):
        A_array[max_row * n + j], A_array[k * n + j] = A_array[k * n + j], A_array[max_row * n + j]
    # division
    denominator = A_array[k * n + k]
    for j in range(k, n):
      A_array[k * n + j] /= denominator
    B_array[k] /= denominator
    # subtraction
    for i in range(k + 1, n):
      multiplier = A_array[i * n + k]
      for j in range(k, n):
        A_array[i * n + j] -= A_array[k * n + j] * multiplier
      B_array[i] -= B_array[k] * multiplier

  # get solution
  B_array[n - 1] /= A_array[(n - 1) * n + n - 1]
  for i in reversed(range(n - 1)):
    for j in range(i + 1, n):
      B_array[i] -= A_array[i * n + j] * B_array[j]


# get result_file
# @param        X_L                 the set of potential causes
# @param        e                   an effect
# @param        B_array             the value vector of the system of linear equations
# @param        result_file         file containing relationships and their causal significance
def get_result_file(X_L, e, B_array, result_file):
  for i in range(len(X_L)):
    c, r, s = X_L[i]
    with open(result_file, 'wb') as f:
      spamwriter = csv.writer(f, delimiter = ',')
      spamwriter.writerow([c, e, r, s, B_array[i]])


# get a linearly independent subset of X
# @param        X_L                 the set of potential causes
# @param        e                   an effect
def get_X_LIS_L(X_L e):
  print "get_X_LIS_L"
  n = len(X_L)
  X_LIS_L = []
  c_L_abs_dif_L = []
  for c_L in X_L:
    c_L_abs_dif_L.append([c_L, abs(get_E_e_c(e, c_L) - get_E_e(e))])
  # bubble sort 
  for i in reversed(range(n - 1)):
    for j in range(i + 1):
      if c_L_abs_dif_L[j][1] < c_L_abs_dif_L[j + 1][1]:
        c_L_abs_dif_L[j], c_L_abs_dif_L[j + 1] = c_L_abs_dif_L[j + 1], c_L_abs_dif_L[j]

  # initialize global variables
  max_row_Dic = {}
  coefficient_Dic = {}
  denominator_Dic = {}
  multiplier_Dic = {}

  # greedy search
  for [c_L, abs_dif] in c_L_abs_dif_L:
    X_LIS_L.append(c_L)
    if len(c_L) > 1:
      A_LIS_array = get_A_array(X_LIS_L)
      if is_full_rank_greedy(A_LIS_array) == False:
        X_LIS_L.remove(c_L)
  return X_LIS_L


# get global_variables: disc_val_Dic, time_cont_val_L_Dic, alphabet_disc, alphabet_cont
# @param        disc_data_file      time series data of form
#                                   var1_t1, var1_t2, ..., var1_tn
#                                   var2_t1, var2_t2, ..., varn_tn
# @param        cont_data_file      time series data of form
#                                   var1_t1, var1_t2, ..., var1_tn
#                                   var2_t1, var2_t2, ..., varn_tn
# @param        header              True,  if there is a header
#                                   False, otherwise
# @param        transpose           tells us whether the data need to be transposed
#                                   False, when the data are of the above form
#                                   True,  when the data are of the following form
#                                   var1_t1, var2_t1, ..., varn_t1
#                                   var1_t2, var2_t2, ..., varn_tn
def get_global_variables(disc_data_file, cont_data_file, header, transpose):
  print "discrete_file: %s" disc_data_file
  print "continuous_file: %s" cont_data_file

  disc_var_time_val_LLL = get_var_time_val_LLL(disc_data_file, header, transpose, "discrete")
  cont_var_time_val_LLL = get_var_time_val_LLL(cont_data_file, header, transpose "continuous")

  # initialize global variables
  relations = {}
  disc_val_Dic = {}
  time_cont_val_Dic = {}
  cont_val_L_Dic = {}

  # get disc_val_Dic
  for [var, time_val_LL] in disc_var_time_val_LLL:
    for [time, val] in time_val_LL:
      if time and val:
        disc_val_Dic[time].append(val)

  # get time_cont_val_L_Dic
  for [var, time_val_LL] in cont_var_time_val_LLL:
    for [time, val] in time_val_LL:
      if time and val:
        time_cont_val_Dic[var].append([time, val])

  # get cont_val_L_Dic
  # for cases where variables are not measured at every timepoint
  for [var, time_val_LL] in cont_var_time_val_LLL:
    for [time, val] in time_val_LL:
      if time and val:
        if not var in cont_val_L_Dic:
          cont_val_L_Dic[var] = {}
        cont_val_L_Dic[var][time] = val

  # get alphabet_disc_Dic
  for time in disc_val_Dic:
    for var in disc_val_Dic[time]:
      if not var in alphabet_disc_Dic:
        alphabet_disc_Dic.append(var)

  # get alphabet_cont_Dic
  for var in time_cont_val_L_Dic:
    alphabet_cont_Dic.append(var)


# get [var, [time, val]]
# @param        time_series_file    time series data of form
#                                   var1_t1, var1_t2, ..., var1_tn
#                                   var2_t1, var2_t2, ..., varn_tn
# @param        header              True,  if there is a header
#                                   False, otherwise
# @param        transpose           tells us whether the data need to be transposed
#                                   False, when the data are of the above form
#                                   True,  when the data are of the following form
#                                   var1_t1, var2_t1, ..., varn_t1
#                                   var1_t2, var2_t2, ..., varn_tn
# @param        data_type           "discrete",   if discrete data
#                                   "continuous", if continuous_valued data
def get_var_time_val_LLL(time_series_file, header, transpose, data_type):
  with open(time_series_file, 'rb') as f:
    spamreader = list(csv.reader(f, delimiter = ','))
    if transpose:
      # transpose the data
      zip(*spamreader)

    var_time_val_LLL = []
    for i in range(len(spamreader)):
      var = ""
      time_val_LL = []

      # get val_L
      val_L = []
      val_start = 0
      if header:
        # the name of the var lies in the first column in each row
        var = spamreader[i][0]
        # the value of the var starts from the second column in each row
        val_start = 1
      else:
        # we use the number of the row as the name of the var in that row
        var = str(i)
      for j in range(val_start, len(spamreader[i])):
        if data_type == "discrete":
          # discrete val is a string, e.g. "High", "Normal", or "Low"
          val_L.append(spamreader[i][j])
        else:
          # continuous val is a number, e.g. 0.5, thus needs to be converted to float
          val_L.append(float(spamreader[i][j]))

      # get time_val_LL
      for time in range(len(val_L)):
        time_val_LL.append([time, val_L[time]])

      # get var_time_val_LLL
      var_time_val_LLL.append([var, time_val_LL])
    return var_time_val_LLL


# generate hypotheses for an effect
# a hypothesis is of form: [cause, effect, window_start, window_end], or [c, e, r, s] for simplicity
# @param        c_L                 [c, r, s], a list including potential cause c, start and end of time window, r and s
# @param        e_L                 [e]
# @param        r                   the start of a time window, i.e. r in window [r, s]
# @param        s                   the end of a time window, i.e. s in window [r, s]
def generate_hypotheses(c_L, e_L, r, s):
  c_e_r_s_L = []
  for e in e_L:
    for c in c_L:
      c_e_r_s_L.append([c, e, r, s])


# test hypotheses
# for hypothesis [c, e, r, s], test whether c is a potential cause of e related to time window [r, s] and get *relations*
# @param        hypotheses          a hypothesis is of form: [c, e, r, s]
# @param        rel_type            type of hypotheses we want to test
                                    "not_equal" for hypotheses s.t. E_e_c != E_e
                                    "positive"  for hypotheses s.t. E_e_c > E_e
                                    "negative"  for hypotheses s.t. E_e_c < E_e
                                    "all"       for all hypotheses
def test_hypotheses(hypotheses, rel_type):
  for [c, e, r, s] in hypotheses:
    if c and e and r and s:
      E_e_c = get_E_e_c([c, r, s])
      E_e = get_E_e(e)
      if E_e_c:
        if rel_type == "not_equal":
          if E_e_c != E_e:
            add_relationship(c, e, r, s)
        elif rel_type == "positive":
          if E_e_c > E_e:
            add_relationship(c, e, r, s)
        elif rel_type == "negative":
          if E_e_c < E_e:
            add_relationship(c, e, r, s)
        elif rel_type == "all":
          add_relationship(c, e, r, s)


# get E[e|c]
# @param        e                   an effect
# @param        c_L                 [c, r, s], a list including potential cause c, start and end of time window, r and s
def get_E_e_c(e, c_L):
  T_e_c_L = get_T_e_c_L(c_L)
  if T_e_c_L:
    if not [e, c_L] in E_e_c_Dic:
      val_L = []
      for time in T_e_c_L:
        val_L.append(cont_val_L_Dic[e][time])
      E_e_c = np.mean(val_L)
      E_e_c_Dic[[e, c_L]] = E_e_c
    return E_e_c_Dic[[e, c_L]]
  else:
    return None

# get E[e]
# @param        e                   an effect
def get_E_e(e):
  if not e in E_e_Dic:
    val_L = []
    for time in cont_val_L_Dic[e]:
      val_L.append(cont_val_L_Dic[e][time])
    E_e = np.mean(val_L)
    E_e_Dic[e] = E_e
  return E_e_Dic[e]


# add relationship [c, e, r, s] to relations_dic
# @param        c                   a potential cause
# @param        e                   an effect
# @param        r                   the start of a time window, i.e. r in window [r, s]
# @param        s                   the end of a time window, i.e. s in window [r, s]
def add_relationship(c, e, r, s):
  relations_dic[e].append([c, r, s])

# main function
if __name__=="__main__":
  if not
