import sys
import os
import csv

import numpy as np
import matplotlib.pyplot as plt
import random


# Data preparation

# global variables

# key: pid->time->var
# val: val
dic = {}

# transform DMITRI data
def transform_DMITRI_data(in_file, out_dir):
  # get dic
  with open(in_file, 'rb') as f:
    spamreader = list(csv.reader(f, delimiter = ','))
    for line in spamreader:
      pid = line[0].strip()
      time = line[1].strip()
      var = line[2].strip()
      val = line[3].strip()
      if not dic.has_key(pid):
        dic[pid] = {}
      if not dic[pid].has_key(time):
        dic[pid][time] = {}
      dic[pid][time][var] = val

  # output individual data
  for pid in dic:
    out_file = out_dir + pid + ".csv"
    with open(out_file, 'wb') as f:
      spamwriter = csv.writer(f, delimiter = ',')

      # output header
      # get var_L
      var_L = []
      for time in sorted(dic[pid]):
        for var in dic[pid][time]:
          if not var in var_L:
            var_L.append(var)
      spamwriter.writerow(var_L)

      # output val
      for time in sorted(dic[pid]):
        val_L = []
        for var in var_L:
          val = None
          if (dic[pid][time].has_key(var)):
            val = dic[pid][time][var]
          val_L.append(val)
        spamwriter.writerow(val_L)


# main function        
if __name__=="__main__":
  if not os.path.exists(sys.argv[2]):
    os.makedirs(sys.argv[2])

  transform_DMITRI_data(sys.argv[1], sys.argv[2])
