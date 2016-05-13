
This is the readme file for dynamic discretization.


# Simulation

# Data generation
python data_generation.py "/Users/jameshuang/Documents/github/research/experiment/dynamic discretization/input/simulated/"

# Discretization
python algorithm.py "/Users/jameshuang/Documents/github/research/experiment/dynamic discretization/input/simulated/" "/Users/jameshuang/Documents/github/research/experiment/dynamic discretization/output/simulated/static/" "/Users/jameshuang/Documents/github/research/experiment/dynamic discretization/output/simulated/dynamic/" "y" "z"


# DMITRI

# Discretization
python algorithm_lb_ub.py "/Users/jameshuang/Documents/server/PHI/PHI_DMITRI/sandbox/james/by_patient/" "/Users/jameshuang/Documents/github/research/experiment/dynamic discretization/output/DMITRI/static/" "/Users/jameshuang/Documents/github/research/experiment/dynamic discretization/output/DMITRI/dynamic/" "insulin" "glu"

# Statistics
python statistics.py "/Users/jameshuang/Documents/github/research/experiment/dynamic discretization/output/DMITRI/static/equal_frequency/log/" "/Users/jameshuang/Documents/github/research/experiment/dynamic discretization/output/DMITRI/static/equal_width/log/" "/Users/jameshuang/Documents/github/research/experiment/dynamic discretization/output/DMITRI/static/KMeans/log/" "/Users/jameshuang/Documents/github/research/experiment/dynamic discretization/output/DMITRI/dynamic/log/"




python statistics.py "/Users/jameshuang/Documents/github/research/experiment/dynamic discretization/output/DMITRI/static/equal_width/log/" "/Users/jameshuang/Documents/github/research/experiment/dynamic discretization/output/DMITRI/dynamic/log/"

python statistics.py "/Users/jameshuang/Documents/github/research/experiment/dynamic discretization/output/DMITRI/static/equal_frequency/log/" "/Users/jameshuang/Documents/github/research/experiment/dynamic discretization/output/DMITRI/dynamic/log/"

python statistics.py "/Users/jameshuang/Documents/github/research/experiment/dynamic discretization/output/DMITRI/static/KMeans/log/" "/Users/jameshuang/Documents/github/research/experiment/dynamic discretization/output/DMITRI/dynamic/log/"



# export python path
export PYTHONPATH=$PYTHONPATH:/System/Library/Frameworks/Python.framework/Versions/2.7/Extras/lib/python


