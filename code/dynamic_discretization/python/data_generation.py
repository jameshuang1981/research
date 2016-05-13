import sys
import os
import csv

import numpy as np
import matplotlib.pyplot as plt
import random


# Dynamic discretization

# global variables

# key: var->time
# val: var's val at time "time"
dic = {}

# number of timepoints
T = 10000


# write time series file
def write_time_series_file(file):
  # toy:
  # z(t + 1) = 100 * x(t) + y(t), where
  # x is in uniform(-1, 1)
  # y is in uniform(-100, 100)

  # write the name of the variables
  with open(file, 'wb') as f:
    spamwriter = csv.writer(f, delimiter=',')
    var_L = ["x", "y", "z"]
    spamwriter.writerow(var_L)

    # initialize dic[var]
    for var in var_L:
      dic[var] = {}

    # set seed
    random.seed(0)

    # generate value at each time
    for time in range(T):
      #val_x = np.random.uniform(-1, 1)
      val_x = 0
      val_y = np.random.uniform(-100, 100)
      val_z = 0

      if time > 1:
        val_z += 100 * dic["x"][time - 1] + dic["y"][time - 1]
      dic["x"][time] = val_x
      dic["y"][time] = val_y
      dic["z"][time] = val_z
      val_L = [val_x, val_y, val_z]
      spamwriter.writerow(val_L)


# main function        
if __name__=="__main__":
  if not os.path.exists(sys.argv[1]):
    os.makedirs(sys.argv[1])

  file = sys.argv[1] + "series_0.csv"
  write_time_series_file(file)

#  # plot figure
#  axes = plt.gca()
#  axes.set_xlim([0,T - 1])
#  axes.set_ylim([-2,2])
#  time_L = [range(T)]
#  plt.plot(time_L)
#  for var in dic:
#    val_L = []
#    for time in range(T):
#      val = dic[var][time]
#      val_L.append(val)
#    val_min = min(val_L)
#    val_max = max(val_L)
#    normalized_val_L = []
#    for val in val_L:
#      normalized_val = -1 + (val - val_min) * 2 / (val_max - val_min)
#      normalized_val_L.append(normalized_val)
#    plt.plot(normalized_val_L, label = var)
#  plt.legend()
#  plt.show()

