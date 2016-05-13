from __future__ import division
import sys
import os
import csv

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

from sklearn import cluster

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
training_end_T = 8000

# start of test
testing_start_T = 8001

# read file and get cont_dic
def read_input_file(file):
  with open(file, 'rb') as f:
    spamreader = list(csv.reader(f, delimiter=','))
    global T
    T = len(spamreader) - 1
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
def static_discretization(tp, spamwritter):
  for var in cont_dic:
    # get val_L
    val_L = []
    for time in range(training_end_T):
      if cont_dic[var].has_key(time):
        val_L.append(cont_dic[var][time])
    val_L = sorted(val_L)
    n = len(val_L)

    # get centroid_lb and centroid_ub
    centroid_lb = 0
    centroid_ub = 0
    if tp == "equal_width":
      min_val = val_L[0]
      max_val = val_L[n - 1]
      centroid_lb = max_val / 3 + 2 * min_val / 3
      centroid_ub = 2 * max_val / 3 + min_val / 3
    elif tp == "equal_frequency":
      idx_lb = int(n / 3)
      idx_ub = int(2 * n / 3)
      centroid_lb = val_L[idx_lb]
      centroid_ub = val_L[idx_ub]
    else:
      pair_LL = []
      for time in range(training_end_T):
        if cont_dic[var].has_key(time):
          pair_L = [cont_dic[var][time], 0]
          pair_LL.append(pair_L)
      kmeans = cluster.KMeans(n_clusters = 2)
      kmeans.fit(pair_LL)
      centroid = kmeans.cluster_centers_
      centroid_lb = min(centroid[0][0], centroid[1][0])
      centroid_ub = max(centroid[0][0], centroid[1][0])

    # output centroid_lb and centroid_ub
    spamwriter.writerow([var, centroid_lb, centroid_ub])

    # get disc_dic[]
    # initialization
    disc_dic[var] = {}
    for time in range(testing_start_T, T):
      if cont_dic[var].has_key(time):
        disc_dic[var][time] = get_discretized_val(var, time, centroid_lb, centroid_ub)


# dynamic discretization
def dynamic_discretization(min_p, max_p, var_c_L, var_e_L, spamwriter):
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

      # get centroid (lowerbound and upperbound) of var_c and var_e that maximizes their cross covariance with time lag p     
      max_cros_cov = None
      max_centroid_c_lb = 0
      max_centroid_c_ub = 0
      max_centroid_e_lb = 0
      max_centroid_e_ub = 0
      max_arg_p = 0

      # get centroid_c_L and centroid_e_L
      centroid_c_L = []
      centroid_e_L = []
      for i in range(1, 10):
        idx_c = int(len(val_c_L) * i / 10)
        idx_e = int(len(val_e_L) * i / 10)
        centroid_c_L.append(val_c_L[idx_c])
        centroid_e_L.append(val_e_L[idx_e])

      print "centroids of %s:" %var_c
      print centroid_c_L
      print "centroids of %s:" %var_e
      print centroid_e_L

      for p in range(min_p, max_p + 1):
        print "p == %d" %p
        for idx_c_lb in range(len(centroid_c_L)):
          for idx_c_ub in range(idx_c_lb + 1, len(centroid_c_L)):
            for idx_e_lb in range(len(centroid_e_L)):
              for idx_e_ub in range(idx_e_lb + 1, len(centroid_e_L)):
                centroid_c_lb = centroid_c_L[idx_c_lb]
                centroid_c_ub = centroid_c_L[idx_c_ub]
                centroid_e_lb = centroid_e_L[idx_e_lb]
                centroid_e_ub = centroid_e_L[idx_e_ub]
                disc_val_c_L = []
                disc_val_e_L = []

                for time in range(training_end_T):
                  if cont_dic[var_c].has_key(time) and cont_dic[var_e].has_key(time + p):
                    disc_val_c = get_discretized_val(var_c, time, centroid_c_lb, centroid_c_ub)
                    disc_val_e = get_discretized_val(var_e, time + p, centroid_e_lb, centroid_e_ub)
                    disc_val_c_L.append(disc_val_c)
                    disc_val_e_L.append(disc_val_e)

                # get cross covariance
                cros_cov = abs(get_cross_covariance(disc_val_c_L, disc_val_e_L))

                # update max_cros_cov and lowerbound/upperbound of centroids
                if max_cros_cov == None or max_cros_cov < cros_cov:
                  max_cros_cov = cros_cov
                  max_centroid_c_lb = centroid_c_lb
                  max_centroid_c_ub = centroid_c_ub
                  max_centroid_e_lb = centroid_e_lb
                  max_centroid_e_ub = centroid_e_ub
                  max_arg_p = p

        print ("max_centroid_c_lb: %f, max_centroid_c_ub: %f, max_centroid_e_lb: %f, max_centroid_e_ub: %f, max_arg_p: %d" %(max_centroid_c_lb, max_centroid_c_ub, max_centroid_e_lb, max_centroid_e_ub, max_arg_p))

      # output centroid
      spamwriter.writerow([max_centroid_c_lb, max_centroid_c_ub, max_centroid_e_lb, max_centroid_e_ub, max_arg_p])

      # get disc_e_c_dic and disc_c_e_dic
      # initialization        
      if not disc_c_e_dic.has_key(var_c):
        disc_c_e_dic[var_c] = {}
      if not disc_c_e_dic[var_c].has_key(var_e):
        disc_c_e_dic[var_c][var_e] = {}
      # initialization 
      if not disc_e_c_dic.has_key(var_e):
        disc_e_c_dic[var_e] = {}
      if not disc_e_c_dic[var_e].has_key(var_c):
        disc_e_c_dic[var_e][var_c] = {}

      for time in range(testing_start_T, T):
        if cont_dic[var_c].has_key(time):
          disc_val_c = get_discretized_val(var_c, time, max_centroid_c_lb, max_centroid_c_ub)
          disc_c_e_dic[var_c][var_e][time] = disc_val_c
        if cont_dic[var_e].has_key(time):
          disc_val_e = get_discretized_val(var_e, time, max_centroid_e_lb, max_centroid_e_ub)
          disc_e_c_dic[var_e][var_c][time] = disc_val_e


# get_centroid_L
def get_centroid_L(var, k):
  # get pair_LL
  pair_LL = []
  for time in sorted(cont_dic[var]):
    pair_L = [cont_dic[var][time], 0]
    pair_LL.append(pair_L)

  # get centroid_L
  centroid_L = []
  kmeans = cluster.KMeans(n_clusters = k)
  kmeans.fit(pair_LL)
  centroid = kmeans.cluster_centers_
  for i in range(len(centroid)):
    if not centroid[i][0] in centroid_L:
      centroid_L.append(centroid[i][0])

  # return sorted centroid_L
  centroid_L.sort()
  return centroid_L


# get_discretized_val     
def get_discretized_val(var, time, centroid_lb, centroid_ub):
  cont_val = cont_dic[var][time]
  if cont_val < centroid_lb:
    return -1
  if cont_val > centroid_ub:
    return 1
  return 0


# calculate cross covariance of lag p      
def get_cross_covariance(val_c_L, val_e_L):
  mean_c = np.mean(val_c_L)
  mean_e = np.mean(val_e_L)
  sum = 0
  for i in range(len(val_c_L)):
    val_c = val_c_L[i]
    val_e = val_e_L[i]
    sum += (val_c - mean_c) * (val_e - mean_e)
  sum /= len(val_c_L)
  return sum


# get precision and recall of static discretization
def get_precision_recall_static(spamwriter):
  # true positive
  tp = 0
  # false positive
  fp = 0
  # false negative
  fn = 0

  # get tp and fp
  for time in range(testing_start_T, T - 1):
    if disc_dic["x"][time] == 1 and disc_dic["y"][time] == 1 or disc_dic["x"][time] == -1 and disc_dic["y"][time] == -1:
      if disc_dic["x"][time] == 1 and disc_dic["y"][time] == 1:
        if cont_dic["x"][time] > 0 and cont_dic["y"][time] > 0 and disc_dic["z"][time + 1] == 1:
          tp += 1
        else:
          fp += 1
      elif disc_dic["x"][time] == -1 and disc_dic["y"][time] == -1:
        if cont_dic["x"][time] < 0 and cont_dic["y"][time] < 0 and disc_dic["z"][time + 1] == -1:
          tp += 1
        else:
          fp += 1

  # get fn
  for time in range(testing_start_T, T - 1):
    if cont_dic["x"][time] > 0 and cont_dic["y"][time] > 0 or cont_dic["x"][time] < 0 and cont_dic["y"][time] < 0:
      if cont_dic["x"][time] > 0 and cont_dic["y"][time] > 0:
        if disc_dic["x"][time] != 1 or disc_dic["y"][time] != 1 or disc_dic["z"][time + 1] != 1:
          fn += 1
      elif cont_dic["x"][time] < 0 and cont_dic["y"][time] < 0:
        if disc_dic["x"][time] != -1 or disc_dic["y"][time] != -1 or disc_dic["z"][time + 1] != -1:
          fn += 1

  # get precision, recall and F_score
  precision = tp / (tp + fp)
  recall = tp / (tp + fn)
  f_score = 2 * precision * recall / (precision + recall)

  # output precision, recall and F_score
  spamwriter.writerow([precision, recall, f_score])

# get precision and recall of dynamic discretization
def get_precision_recall_dynamic(spamwriter):
  # precision = tp / (tp + fp)
  # recall = tp / (tp + fn)
  # true positive
  tp = 0
  # false positive
  fp = 0
  # false negative
  fn = 0

  # get tp and fp
  for time in range(testing_start_T, T - 1):
    if disc_c_e_dic["x"]["z"][time] == 1 and disc_c_e_dic["y"]["z"][time] == 1 or disc_c_e_dic["x"]["z"][time] == -1 and disc_c_e_dic["y"]["z"][time] == -1:
      if disc_c_e_dic["x"]["z"][time] == 1 and disc_c_e_dic["y"]["z"][time] == 1:
        if cont_dic["x"][time] > 0 and cont_dic["y"][time] > 0 and disc_e_c_dic["z"]["x"][time + 1] == 1 and disc_e_c_dic["z"]["y"][time + 1] == 1:
          tp += 1
        else:
          fp += 1
      elif disc_c_e_dic["x"]["z"][time] == -1 and disc_c_e_dic["y"]["z"][time] == -1:
        if cont_dic["x"][time] < 0 and cont_dic["y"][time] < 0 and disc_e_c_dic["z"]["x"][time + 1] == -1 and disc_e_c_dic["z"]["y"][time + 1] == -1:
          tp += 1
        else:
          fp += 1

  # get fn
  for time in range(testing_start_T, T - 1):
    if cont_dic["x"][time] > 0 and cont_dic["y"][time] > 0 or cont_dic["x"][time] < 0 and cont_dic["y"][time] < 0:
      if cont_dic["x"][time] > 0 and cont_dic["y"][time] > 0:
        if disc_c_e_dic["x"]["z"][time] != 1 or disc_c_e_dic["y"]["z"][time] != 1 or disc_e_c_dic["z"]["x"][time + 1] != 1 or disc_e_c_dic["z"]["y"][time + 1] != 1:
          fn += 1
      elif cont_dic["x"][time] < 0 and cont_dic["y"][time] < 0:
        if disc_c_e_dic["x"]["z"][time] != -1 or disc_c_e_dic["y"]["z"][time] != -1 or disc_e_c_dic["z"]["x"][time + 1] != -1 or disc_e_c_dic["z"]["y"][time + 1] != -1:
          fn += 1

  # get precision, recall and F_score
  precision = tp / (tp + fp)
  recall = tp / (tp + fn)
  f_score = 2 * precision * recall / (precision + recall)

  # output precision, recall and F_score
  spamwriter.writerow([precision, recall, f_score])


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

      # output precision, recall and F-score of discretization based on equal width
      with open(sys.argv[2] + "equal_width/log/" + file, 'wb') as f:
        spamwriter = csv.writer(f, delimiter = ',')
        static_discretization("equal_width", spamwriter)
      with open(sys.argv[2] + "equal_width/statistics/" + file, 'wb') as f:
        spamwriter = csv.writer(f, delimiter = ',')
        get_precision_recall_static(spamwriter)

      # output precision, recall and F-score of discretization based on equal frequency
      with open(sys.argv[2] + "equal_frequency/log/" + file, 'wb') as f:
        spamwriter = csv.writer(f, delimiter = ',')
        static_discretization("equal_frequency", spamwriter)
      with open(sys.argv[2] + "equal_frequency/statistics/" + file, 'wb') as f:
        spamwriter = csv.writer(f, delimiter = ',')
        get_precision_recall_static(spamwriter)

      # output precision, recall and F-score of discretization based on KMeans
      with open(sys.argv[2] + "KMeans/log/" + file, 'wb') as f:
        spamwriter = csv.writer(f, delimiter = ',')
        static_discretization("KMeans", spamwriter)
      with open(sys.argv[2] + "KMeans/statistics/" + file, 'wb') as f:
        spamwriter = csv.writer(f, delimiter = ',')
        get_precision_recall_static(spamwriter)

      # output precision, recall and F-score of dynamic discretization
      with open(sys.argv[3] + "log/" + file, 'wb') as f:
        spamwriter = csv.writer(f, delimiter = ',')
        dynamic_discretization(1, 5, var_c_L, var_e_L, spamwriter)
      with open(sys.argv[3] + "statistics/" + file, 'wb') as f:
        spamwriter = csv.writer(f, delimiter = ',')
        get_precision_recall_dynamic(spamwriter)
