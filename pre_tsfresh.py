'''
Use this code to extract the instance-level meta-features for a single dataset.
In the github repo, we made available a .sh to run all the datasets at once.
'''

import argparse
import numpy as np
import pandas as pd
import sys

from numpy import genfromtxt
from tsfresh.feature_extraction import extract_features, MinimalFCParameters

def parser_args(cmd_args):

  parser = argparse.ArgumentParser(sys.argv[0], description="",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("-d", "--dataset", type=str, action="store", default="",
    help="Dataset name (considering it is in the 'data' folder)")
  
  return parser.parse_args(cmd_args)

args = parser_args(sys.argv[1:])
data_name = args.dataset

print("Starting the preprocessing of " + data_name + " dataset with TSFRESH")

'''
Training dataset
'''
print("Loading training dataset")

file_path = "./data/" + data_name + "/" + data_name + "_TRAIN.tsv"
data = genfromtxt(file_path, delimiter='\t')

teste = []

i = 0
for ts in data:
	for obs in ts[1:]:
		teste.append([i,obs,ts[0]])
	i += 1

print("Done")
print("Extracting features")

cols = ["id","x","y"]
df = pd.DataFrame(teste,columns=cols)

X = extract_features(df, default_fc_parameters=MinimalFCParameters(), column_id='id')
X.to_csv("./tsfresh/" + data_name + "_TRAIN.csv")
print("Done!\n")

