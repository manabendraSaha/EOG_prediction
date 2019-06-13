import tensorflow as tf

import pandas as pd
import os
import glob
import numpy

training_folder = 'C:\\projects\\anirban\\dataset'
os.chdir(training_folder)
files = glob.glob('*.csv')

for file in files:
    print(file)
    path = os.path.join(training_folder, file)
    df1 = pd.read_csv(path)

source = 'C:\\projects\\anirban\\dataset\\part1.csv'

dataframe = pd.read_csv(source)

sample = dataframe.head()
print(sample)

feature_set = dataframe[['ACC_X','ACC_Y','ACC_Z', 'GYRO_X', 'GYRO_Y', 'GYRO_Z']]
gt_set = dataframe[['EOG_L', 'EOG_R', 'EOG_H', 'EOG_V']]

a_x_max = feature_set['ACC_X'].abs().max()
a_y_max = feature_set['ACC_Y'].abs().max()
a_z_max =feature_set['ACC_Z'].abs().max()
g_x_max =feature_set['GYRO_X'].abs().max()
g_y_max =feature_set['GYRO_Y'].abs().max()
g_z_max =feature_set['GYRO_Z'].abs().max()

feature_set['ACC_X'] = feature_set['ACC_X'].div(a_x_max)
feature_set['ACC_Y'] = feature_set['ACC_Y'].div(a_y_max)
feature_set['ACC_Z'] = feature_set['ACC_Z'].div(a_z_max)

print(feature_set['ACC_X'].abs().max())