import numpy as np
import nibabel as nib
import os
import glob
import pandas as pd

df=pd.read_csv("./prediction/brats_scores.csv")

f = open("./prediction/prediction_eval.txt",'w')

print("DICE WholeTumor mean: ",df['WholeTumor'].mean(), " DICE WholeTumor std: ",df['WholeTumor'].std())
str1 = "DICE WholeTumor mean: " + str(df['WholeTumor'].mean()) + " DICE WholeTumor std: " + str(df['WholeTumor'].std()) + "\n"
f.write(str1)

print("DICE TumorCore mean: ",df['TumorCore'].mean(), " DICE TumorCore std: ",df['TumorCore'].std())
str1 = "DICE TumorCore mean: " + str(df['TumorCore'].mean()) + " DICE TumorCore std: " + str(df['TumorCore'].std()) + "\n"
f.write(str1)

print("DICE EnhancingTumor mean: ",df['EnhancingTumor'].mean(), " DICE EnhancingTumor std: ",df['EnhancingTumor'].std())
str1 = "DICE EnhancingTumor mean: " + str(df['EnhancingTumor'].mean()) + " DICE EnhancingTumor std: " + str(df['EnhancingTumor'].std()) + "\n"
f.write(str1)

f.close()
