import tensorflow as tf

import pandas as pd
import os
import glob
import numpy


training_folder = 'training folder location'
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


#print(dataframe['ACC_X'].max())
#print(dataframe['ACC_X'].min())

feature_arr = numpy.asarray(feature_set)
gt_arr = numpy.asarray(gt_set)
filtered_gt = gt_arr[49:,:]
print(filtered_gt.shape)
print(feature_arr.shape)

height, width = feature_arr.shape
filtered_feature = numpy.zeros((height-49, 300), dtype=numpy.float32)
i = 0
while i < height - 49:
    filtered_feature[i,:] = feature_arr[i : 50 + i, :].reshape(1,300)
    i += 1

print(filtered_feature.shape)


model_directory = 'location_for_model'
import modelInterface
modelInterface.train_encoder(filtered_feature, filtered_gt)