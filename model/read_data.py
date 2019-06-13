
import pandas as pd
import numpy
import os


train_location = 'location_for_training_file'

from glob import glob
os.chdir(train_location)
train_files = glob('*.csv')
final_feature_array = []
final_gt_array = []
for counter, file  in enumerate(train_files):
    print(os.path.join(train_location, file))
    whole_dataframe = pd.read_csv(os.path.join(train_location, file))
    feature_set = whole_dataframe[['ACC_X','ACC_Y','ACC_Z', 'GYRO_X', 'GYRO_Y', 'GYRO_Z']]
    feature_set = numpy.asarray(feature_set)
    gt_set = whole_dataframe[['EOG_L', 'EOG_R', 'EOG_H', 'EOG_V']]
    gt_set = numpy.asarray(gt_set)

    ##implement sliding window for the collected dataset
    height, width = whole_dataframe.shape
    filtered_feature = []
    filtered_gt = []


    if(counter == 0):
        final_feature_array = numpy.zeros((height - 49, 300), dtype=numpy.float32)
        final_gt_array = numpy.zeros((height - 49, 4), dtype=numpy.float32)
        final_gt_array = numpy.copy(gt_set[49:, :])
    else:
        filtered_feature = numpy.zeros((height - 49, 300), dtype=numpy.float32)
        filtered_gt = numpy.zeros((height - 49, 4), dtype=numpy.float32)
        filtered_gt = numpy.copy(gt_set[49:, :])


    i = 0
    while i < height - 49:
        if(counter == 0):
            final_feature_array[i, :] = numpy.copy(feature_set[i: 50 + i, :]).reshape(1, 300)
        else:
            filtered_feature[i, :] = numpy.copy(feature_set[i: 50 + i, :]).reshape(1, 300)
        i += 1

    if(counter != 0):
        final_feature_array = numpy.vstack((final_feature_array, filtered_feature))
        final_gt_array = numpy.vstack((final_gt_array, filtered_gt))

    #print(final_feature_array.shape)
    #print(final_gt_array.shape)


model_directory = 'model_directory'
import main as model
#model.train_encoder(final_feature_array, final_gt_array, model_directory, epochs = 20, train_test_split = 0.90)

model_location = 'location_to_model\\17_ckpt\\17.ckpt'

out_dir = 'output_location\\output\\predicted'
#for feature in final_feature_array:
model.predict_result(model_location = model_location, feature_set=final_feature_array, gt_set = final_gt_array, output_directory = out_dir)

