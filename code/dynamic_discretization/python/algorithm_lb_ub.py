from __future__ import division
import sys
import os
import csv

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn import cluster
from scipy.stats.stats import pearsonr

# Dynamic discretization

# global variables

# key: var->time
# val: var's val at time "time"
cont_dic = {}
disc_dic = {}

# key: var_e->var_c->time
# val: var_e's val related to var_c at time t 
disc_e_c_dic = {}

# key: var_c->var_e->time
# val: var_c's val related to var_e at time t 
disc_c_e_dic = {}

# number of timepoints
T = 0

# end of training
training_end_T = 0

# start of test
testing_start_T = 0

# key: var_c->var_e
# val: time lag p that yields the maximum cross correlation between var_c and var_e
argmax_p_dic = {}

# read file and get cont_dic
def read_input_file(file):
  with open(file, 'rb') as f:
    spamreader = list(csv.reader(f, delimiter=','))
    global T
    T = len(spamreader) - 1
    global training_end_T
    training_end_T = int(8 * T / 10)
    global testing_start_T
    testing_start_T = training_end_T + 1
    for time in range(T):
      for var_idx in range(len(spamreader[0])):
        var = spamreader[0][var_idx].strip()
        val = spamreader[time + 1][var_idx]
        if val:
          if not cont_dic.has_key(var):
            cont_dic[var] = {}
          cont_dic[var][time] = float(val)


# static discretization
# tp == "equal_width" when using equal width discretization
# tp == "equal_frequency" when using equal frequency discretization
# tp == "KMeans" when using KMeans
def static_discretization(min_p, max_p, var_c_L, var_e_L, tp, spamwritter):
  # initialize disc_dic
  disc_dic = {}
  for var in cont_dic:
    if not var in var_c_L and not var in var_e_L:
      continue
    # initialize disc_dic[var]
    if not disc_dic.has_key(var):
      disc_dic[var] = {}

    # get val_L
    val_L = []
    for time in range(T):
      if cont_dic[var].has_key(time):
        val_L.append(cont_dic[var][time])
    val_L = sorted(val_L)
    n = len(val_L)
    # get lowerbound and upperbound
    lowerbound = 0
    upperbound = 0
    if tp == "equal_width":
      min_val = val_L[0]
      max_val = val_L[n - 1]
      lowerbound = max_val / 3 + 2 * min_val / 3
      upperbound = 2 * max_val / 3 + min_val / 3
    elif tp == "equal_frequency":
      idx_lb = int(n / 3)
      idx_ub = int(2 * n / 3)
      lowerbound = val_L[idx_lb]
      upperbound = val_L[idx_ub]
    else:
      pair_LL = []
      for time in range(T):
        if cont_dic[var].has_key(time):
          pair_L = [cont_dic[var][time], 0]
          pair_LL.append(pair_L)
      kmeans = cluster.KMeans(n_clusters = 2)
      kmeans.fit(pair_LL)
      centroid = kmeans.cluster_centers_
      lowerbound = min(centroid[0][0], centroid[1][0])
      upperbound = max(centroid[0][0], centroid[1][0])

    # output lowerbound and upperbound
    spamwriter.writerow([var, lowerbound, upperbound])

    # discretize data based on lowerbound and upperbound
    for time in range(T):
      if cont_dic[var].has_key(time):
        cont_val = cont_dic[var][time]
        disc_val = get_discretized_val(var, time, lowerbound, upperbound)
        disc_dic[var][time] = disc_val

  # get cross correlation related to argmax_p
  for var_c in var_c_L:
    for var_e in var_e_L:
      disc_val_c_L = []
      disc_val_e_L = []
      for time in range(T):
        if disc_dic[var_c].has_key(time) and disc_dic[var_e].has_key(time + argmax_p_dic[var_c][var_e]):
          disc_val_c = disc_dic[var_c][time]
          disc_val_e = disc_dic[var_e][time + argmax_p_dic[var_c][var_e]]
          disc_val_c_L.append(disc_val_c)
          disc_val_e_L.append(disc_val_e)

      # get cross correlation
      cros_corr = pearsonr(disc_val_c_L, disc_val_e_L)
      cros_corr = cros_corr[0]

      # output max cross correlation
      spamwriter.writerow([var_c, var_e, cros_corr, argmax_p_dic[var_c][var_e]])


# dynamic discretization
def dynamic_discretization(min_p, max_p, bin_num, var_c_L, var_e_L, spamwriter):
  for var_c in var_c_L:
    # get val_c_L
    val_c_L = []
    for time in range(T):
      if cont_dic[var_c].has_key(time):
        val_c_L.append(cont_dic[var_c][time])
    val_c_L = sorted(val_c_L)

    for var_e in var_e_L:
      # get val_e_L
      val_e_L = []
      for time in range(T):
        if cont_dic[var_e].has_key(time):
          val_e_L.append(cont_dic[var_e][time])
      val_e_L = sorted(val_e_L)

      # get max cross correlation, the corresponding lowerbound and upperbound of var_c and var_e
      max_cros_corr = None
      max_lb_c = 0
      max_ub_c = 0
      max_lb_e = 0
      max_ub_e = 0
      argmax_p = 0

      bound_c_L = []
      bound_e_L = []

      for i in range(1, bin_num):
        idx_c = int(len(val_c_L) * i / bin_num)
        idx_e = int(len(val_e_L) * i / bin_num)
        bound_c_L.append(val_c_L[idx_c])
        bound_e_L.append(val_e_L[idx_e])

#       print "bound_c_L of %s:" %var_c
#       print bound_c_L
#       print "bound_e_L of %s:" %var_e
#       print bound_e_L

      for p in range(min_p, max_p + 1):
        # print "p == %d" %p
        for idx_c_lb in range(len(bound_c_L)):
          for idx_c_ub in range(idx_c_lb + 1, len(bound_c_L)):
            for idx_e_lb in range(len(bound_e_L)):
              for idx_e_ub in range(idx_e_lb + 1, len(bound_e_L)):
                lb_c = bound_c_L[idx_c_lb]
                ub_c = bound_c_L[idx_c_ub]
                lb_e = bound_e_L[idx_e_lb]
                ub_e = bound_e_L[idx_e_ub]

                disc_val_c_L = []
                disc_val_e_L = []

                for time in range(T):
                  if cont_dic[var_c].has_key(time) and cont_dic[var_e].has_key(time + p):
                    disc_val_c = get_discretized_val(var_c, time, lb_c, ub_c)
                    disc_val_e = get_discretized_val(var_e, time + p, lb_e, ub_e)
                    disc_val_c_L.append(disc_val_c)
                    disc_val_e_L.append(disc_val_e)

                # get cross correlation
                cros_corr = pearsonr(disc_val_c_L, disc_val_e_L)
                cros_corr = cros_corr[0]

                # update max_cros_corr and lowerbound/upperbound of var_c and var_e 
                if max_cros_corr == None or abs(max_cros_corr) < abs(cros_corr):
                  max_cros_corr = cros_corr
                  max_lb_c = lb_c
                  max_ub_c = ub_c
                  max_lb_e = lb_e
                  max_ub_e = ub_e
                  argmax_p = p

                # print ("p: %f, max_cros_corr: %f, max_lb_c: %f, max_ub_c: %f, max_lb_e: %f, max_ub_e: %f" %(p, max_cros_corr, max_lb_c, max_ub_c, max_lb_e, max_ub_e))

      # get argmax_p_dic[var_c][var_e]
      if not argmax_p_dic.has_key(var_c):
        argmax_p_dic[var_c] = {}
      argmax_p_dic[var_c][var_e] = argmax_p

      # output arguments
      spamwriter.writerow([var_c, var_e, argmax_p, max_cros_corr, max_lb_c, max_ub_c, max_lb_e, max_ub_e])

      # discretize data based on max_lb_c, max_ub_c, max_lb_e, and max_ub_e
      # initialization
      if not disc_c_e_dic.has_key(var_c):
        disc_c_e_dic[var_c] = {}
      if not disc_c_e_dic[var_c].has_key(var_e):
        disc_c_e_dic[var_c][var_e] = {}
      if not disc_e_c_dic.has_key(var_c):
        disc_e_c_dic[var_e] = {}
      if not disc_e_c_dic[var_e].has_key(var_c):
        disc_e_c_dic[var_e][var_c] = {}

      for time in range(T):
        if cont_dic[var_c].has_key(time):
          disc_val_c = get_discretized_val(var_c, time, max_lb_c, max_ub_c)
          disc_c_e_dic[var_c][var_e][time] = disc_val_c
        if cont_dic[var_e].has_key(time + argmax_p):
          disc_val_e = get_discretized_val(var_e, time + argmax_p, max_lb_e, max_ub_e)
          disc_e_c_dic[var_e][var_c][time + argmax_p] = disc_val_e

      # get max cross correlation
      cont_val_c_L = []
      cont_val_e_L = []
      for time in range(T):
        if cont_dic[var_c].has_key(time) and cont_dic[var_e].has_key(time + argmax_p):
          cont_val_c = cont_dic[var_c][time]
          cont_val_e = cont_dic[var_e][time + argmax_p]
          cont_val_c_L.append(cont_val_c)
          cont_val_e_L.append(cont_val_e)

      # get cross correlation
      cros_corr = pearsonr(cont_val_c_L, cont_val_e_L)
      cros_corr = cros_corr[0]

      # output max cross correlation
      spamwriter.writerow([var_c, var_e, argmax_p, cros_corr])


# get_discretized_val     
def get_discretized_val(var, time, lowerbound, upperbound):
  cont_val = cont_dic[var][time]
  if cont_val < lowerbound:
    return -1
  if cont_val > upperbound:
    return 1
  return 0


# get cross correlation
def get_cros_corr(var_c_L, var_e_L, spamwriter, tp):
  for var_c in var_c_L:
    for var_e in var_e_L:
      p = argmax_p_dic[var_c][var_e]
      # get val_c_L and val_e_L
      val_c_L = []
      val_e_L = []
      for time in range(testing_start_T, T - p):
        if cont_dic[var_c].has_key(time) and cont_dic[var_e].has_key(time + p):
          if tp == "static":
            val_c_L.append(disc_dic[var_c][time])
            val_e_L.append(disc_dic[var_e][time + p])
          else:
            val_c_L.append(disc_c_e_dic[var_c][var_e][time])
            val_e_L.append(disc_e_c_dic[var_e][var_c][time + p])
      # get cross correlation
      cros_corr = pearsonr(val_c_L, val_e_L)
      cros_corr = cros_corr[0]
      spamwriter.writerow([var_c, var_e, cros_corr])


# main function        
if __name__=="__main__":
  if not os.path.exists(sys.argv[2] + "equal_width/log/"):
    os.makedirs(sys.argv[2] + "equal_width/log/")
  if not os.path.exists(sys.argv[2] + "equal_frequency/log/"):
    os.makedirs(sys.argv[2] + "equal_frequency/log/")
  if not os.path.exists(sys.argv[2] + "KMeans/log/"):
    os.makedirs(sys.argv[2] + "KMeans/log/")
  if not os.path.exists(sys.argv[3] + "log/"):
    os.makedirs(sys.argv[3] + "log/")

  if not os.path.exists(sys.argv[2] + "equal_width/statistics/"):
    os.makedirs(sys.argv[2] + "equal_width/statistics/")
  if not os.path.exists(sys.argv[2] + "equal_frequency/statistics/"):
    os.makedirs(sys.argv[2] + "equal_frequency/statistics/")
  if not os.path.exists(sys.argv[2] + "KMeans/statistics/"):
    os.makedirs(sys.argv[2] + "KMeans/statistics/")
  if not os.path.exists(sys.argv[3] + "statistics/"):
    os.makedirs(sys.argv[3] + "statistics/")

  for file in os.listdir(sys.argv[1]):
    if file.endswith(".csv"):
      print file
      var_c_L = sys.argv[4].split()
      var_e_L = sys.argv[5].split()
      read_input_file(sys.argv[1] + file)

      # output precision, recall and F-score of dynamic discretization
      with open(sys.argv[3] + "log/" + file, 'wb') as f:
        spamwriter = csv.writer(f, delimiter = ',')
        dynamic_discretization(1, 36, 20, var_c_L, var_e_L, spamwriter)

      # output precision, recall and F-score of discretization based on equal width
      with open(sys.argv[2] + "equal_width/log/" + file, 'wb') as f:
        spamwriter = csv.writer(f, delimiter = ',')
        static_discretization(1, 36, var_c_L, var_e_L, "equal_width", spamwriter)

      # output precision, recall and F-score of discretization based on equal frequency
      with open(sys.argv[2] + "equal_frequency/log/" + file, 'wb') as f:
        spamwriter = csv.writer(f, delimiter = ',')
        static_discretization(1, 36, var_c_L, var_e_L, "equal_frequency", spamwriter)

      # output precision, recall and F-score of discretization based on KMeans
      with open(sys.argv[2] + "KMeans/log/" + file, 'wb') as f:
        spamwriter = csv.writer(f, delimiter = ',')
        static_discretization(1, 36, var_c_L, var_e_L, "KMeans", spamwriter)

