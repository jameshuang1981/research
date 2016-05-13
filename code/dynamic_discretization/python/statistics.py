import sys
import os
import csv

import numpy as np
import matplotlib.pyplot as plt
import random



# output statistics
def output_statistics_log_summary(stat_dir):
  stat_sum_dir = stat_dir + "sum/"
  if not os.path.exists(stat_sum_dir):
    os.makedirs(stat_sum_dir)

  cros_corr_L = []
  lag_L = []

  for file in os.listdir(stat_dir):
    if file.endswith(".csv"):
      with open(stat_dir + file, "rb") as f:
        spamreader = list(csv.reader(f, delimiter = ','))
        if not spamreader[0][2] == "nan" and not spamreader[0][2] == "False":
          cros_corr_L.append(float(spamreader[0][2].strip()))
        if not spamreader[0][3] == "nan" and not spamreader[0][3] == "False":
          lag_L.append(float(spamreader[0][3].strip()))

  with open(stat_sum_dir + "summary.txt", "wb") as f:
    spamwriter = csv.writer(f, delimiter = ',')
    spamwriter.writerow(["cross_correlation_mean", "cross_correlation_std", "lag_mean", "lag_std"])
    spamwriter.writerow([np.mean(cros_corr_L), np.std(cros_corr_L), np.mean(lag_L), np.std(lag_L)])


# output statistics
def output_statistics_summary(stat_dir, tp):
  stat_sum_dir = stat_dir + "sum/"
  if not os.path.exists(stat_sum_dir):
    os.makedirs(stat_sum_dir)

  lb_L = []
  ub_L = []
  cros_corr_disc_L = []
  cros_corr_cont_L = []

  for file in os.listdir(stat_dir):
    if file.endswith(".csv"):
      with open(stat_dir + file, "rb") as f:
        spamreader = list(csv.reader(f, delimiter = ','))
        if tp == "static":
          lb_L.append(float(spamreader[1][1].strip()))
          ub_L.append(float(spamreader[1][2].strip()))
          cros_corr_disc_L.append(float(spamreader[2][2].strip()))
        else:
          lb_L.append(float(spamreader[0][6].strip()))
          ub_L.append(float(spamreader[0][7].strip()))
          cros_corr_disc_L.append(float(spamreader[0][3].strip()))
          cros_corr_cont_L.append(float(spamreader[1][3].strip()))

  with open(stat_sum_dir + "summary.txt", "wb") as f:
    spamwriter = csv.writer(f, delimiter = ',')
    spamwriter.writerow(["lb_mean", "lb_std", "ub_mean", "ub_std"])
    spamwriter.writerow([np.mean(lb_L), np.std(lb_L), np.mean(ub_L), np.std(ub_L)])
    spamwriter.writerow([np.mean(cros_corr_disc_L), np.std(cros_corr_disc_L)])
    if tp == "dynamic":
      spamwriter.writerow([np.mean(cros_corr_cont_L), np.std(cros_corr_cont_L)])


# get statistics of static related results
def get_statistics_static():
  stat_dir = sys.argv[1] + "statistics/"
  if not os.path.exists(stat_dir):
    os.makedirs(stat_dir)

  centroid_lb_L = []
  centroid_ub_L = []

  for file in os.listdir(sys.argv[1]):
    if file.endswith(".csv"):
      with open(sys.argv[1] + file, "rb") as f:
        spamreader = list(csv.reader(f, delimiter = ','))
        centroid_lb_L.append(float(spamreader[9][1].strip()))
        centroid_ub_L.append(float(spamreader[9][2].strip()))

        centroid_lb_mean = np.mean(centroid_lb_L)
        centroid_lb_std = np.std(centroid_lb_L)
        centroid_ub_mean = np.mean(centroid_ub_L)
        centroid_ub_std = np.std(centroid_ub_L)

        with open(stat_dir + "statistics.csv", "wb") as f:
          spamwriter = csv.writer(f, delimiter = ',')
          spamwriter.writerow(["mean", "std"])
          spamwriter.writerow([centroid_lb_mean, centroid_lb_std])
          spamwriter.writerow([centroid_ub_mean, centroid_ub_std])


def get_statistics_dynamic():
  stat_dir = sys.argv[2] + "statistics/"
  if not os.path.exists(stat_dir):
    os.makedirs(stat_dir)

  centroid_lb_dic = {}
  centroid_ub_dic = {}

  for file in os.listdir(sys.argv[1]):
    if file.endswith(".csv"):
      with open(sys.argv[2] + file, "rb") as f:
        spamreader = list(csv.reader(f, delimiter = ','))
        for i in range(len(spamreader)):
          var_c = spamreader[i][0].strip()
          var_e = spamreader[i][1].strip()
          centroid_lb = float(spamreader[i][4].strip())
          centroid_ub = float(spamreader[i][5].strip())
          if not centroid_lb_dic.has_key(var_c):
            centroid_lb_dic[var_c] = {}
          if not centroid_lb_dic[var_c].has_key(var_e):
            centroid_lb_dic[var_c][var_e] = []
          centroid_lb_dic[var_c][var_e].append(centroid_lb)

          if not centroid_ub_dic.has_key(var_c):
            centroid_ub_dic[var_c] = {}
          if not centroid_ub_dic[var_c].has_key(var_e):
            centroid_ub_dic[var_c][var_e] = []
          centroid_ub_dic[var_c][var_e].append(centroid_ub)

  with open(stat_dir + "statistics.csv", "wb") as f:
    spamwriter = csv.writer(f, delimiter = ',')
    spamwriter.writerow(["var_c", "var_e", "centroid_lb_mean", "centroid_ub_mean", "centroid_lb_std", "centroid_ub_std"])

    for var_c in centroid_lb_dic:
      for var_e in centroid_lb_dic[var_c]:
        centroid_lb_mean = np.mean(centroid_lb_dic[var_c][var_e])
        centroid_lb_std = np.std(centroid_lb_dic[var_c][var_e])
        centroid_ub_mean = np.mean(centroid_ub_dic[var_c][var_e])
        centroid_ub_std = np.std(centroid_ub_dic[var_c][var_e])
        spamwriter.writerow([var_c, var_e, centroid_lb_mean, centroid_ub_mean, centroid_lb_std, centroid_ub_std])


# get statistics of dynamic related results

# main function        
if __name__=="__main__":
  output_statistics_summary(sys.argv[1], "static")
  output_statistics_summary(sys.argv[2], "static")
  output_statistics_summary(sys.argv[3], "static")
  output_statistics_summary(sys.argv[4], "dynamic")
  # output_statistics_log_summary(sys.argv[5])
  #get_statistics_static()
  #get_statistics_dynamic()


