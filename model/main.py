import tensorflow as tf
import model
import os
import numpy
import datetime
import pandas as pd
from matplotlib import pyplot as plt


def create_directory(full_file_path):
    try:
        if not os.path.exists(full_file_path):
            os.mkdir(full_file_path)
            return full_file_path
    except RuntimeError:
        return 'None'


def create_folder(path, folder_name):
    folder_path = os.path.join(path, folder_name)
    try:
        return create_directory(folder_path)
    except RuntimeError:
        return 'None'


def save_checkpoint(sess, saver, directory, checkpoint_number):
    try:
        directory_name = str(checkpoint_number) + '_ckpt'
        fileName = str(checkpoint_number) + '.ckpt'
        file_directory = os.path.join(directory, directory_name)
        file_save_directory = os.path.join(file_directory, fileName)
        create_folder(directory, directory_name)
        saver.save(sess, file_save_directory)
        return True
    except Exception:
        print('Exception : in saving check point' )
        return None



def create_session(saved_model_location):
    saver = tf.train.Saver(max_to_keep=500)
    sess = tf.Session()
    saver.restore(sess, saved_model_location)
    return sess, saver


def predict_result(model_location, feature_set, gt_set, output_directory = ''):
    sess, saver = create_session(model_location)
    dummy_gt = numpy.zeros((1,4))
    #data = {'EOG_L':[], 'EOG_R':[], 'EOG_H':[], 'EOG_V':[] }
    symbols = ['EOG_L', 'EOG_R', 'EOG_H', 'EOG_V']
    df = pd.DataFrame(columns= symbols)
    df = pd.DataFrame()
    arr = numpy.zeros((1,4))
    gt_arr = numpy.zeros((1,4))
    diff_arr = numpy.zeros((1,4))
    EOG_L_plot = numpy.zeros((1,2))
    EOG_R_plot = numpy.zeros((1, 2))
    EOG_H_plot = numpy.zeros((1, 2))
    EOG_V_plot = numpy.zeros((1, 2))

    for counter, feature in enumerate(feature_set):
        feature = feature.reshape(1, 300)
        predicted = sess.run(model.output_layer, feed_dict={model.dataset_holder: feature, model.gt_holder : dummy_gt})
        gt = gt_set[counter]
        if(counter == 0):
            arr[0,:] = predicted[0]
            difference = numpy.abs(predicted[0] - gt)
            diff_arr[0,:] = difference
            gt_arr[0,:] = gt
            EOG_L_plot[0,0] = gt[0]
            EOG_L_plot[0,1] = predicted[0,0]
            EOG_R_plot[0, 0] = gt[1]
            EOG_R_plot[0, 1] = predicted[0, 1]
            EOG_H_plot[0, 0] = gt[2]
            EOG_H_plot[0, 1] = predicted[0, 2]
            EOG_V_plot[0, 0] = gt[3]
            EOG_V_plot[0, 1] = predicted[0, 3]
        else:
            arr = numpy.vstack((arr, predicted[0]))
            difference = numpy.abs(predicted[0] - gt)
            diff_arr = numpy.vstack((diff_arr, difference))
            gt_arr = numpy.vstack((gt_arr, gt))
            EOG_L_plot = numpy.vstack((EOG_L_plot, [gt[0], predicted[0,0]]))
            EOG_R_plot = numpy.vstack((EOG_R_plot, [gt[1], predicted[0, 1]]))
            EOG_H_plot = numpy.vstack((EOG_H_plot, [gt[2], predicted[0, 2]]))
            EOG_V_plot = numpy.vstack((EOG_V_plot, [gt[3], predicted[0, 3]]))

    numpy.savetxt(os.path.join(output_directory, 'predicted.csv'), arr)
    numpy.savetxt(os.path.join(output_directory, 'ground_truth.csv'), gt_arr)
    numpy.savetxt(os.path.join(output_directory, 'difference.csv'), diff_arr)
    EOG_L_df = pd.DataFrame(EOG_L_plot, columns=['ground truth', 'predicted'])
    EOG_R_df = pd.DataFrame(EOG_R_plot, columns=['ground truth', 'predicted'])
    EOG_H_df = pd.DataFrame(EOG_H_plot, columns=['ground truth', 'predicted'])
    EOG_V_df = pd.DataFrame(EOG_V_plot, columns=['ground truth', 'predicted'])
    ax1 = EOG_L_df.plot.scatter(x='ground truth',y = 'predicted',c = 'DarkBlue')
    ax2 = EOG_R_df.plot.scatter(x='ground truth', y='predicted', c='DarkBlue')
    ax3 = EOG_H_df.plot.scatter(x='ground truth', y='predicted', c='DarkBlue')
    ax4 = EOG_V_df.plot.scatter(x='ground truth', y='predicted', c='DarkBlue')
    plt.show()








def train_encoder(feature_array, gt_array, output_directory, epochs = 10, train_test_split = 0.75):
    #directory = 'C:\\projects\\anirban\\output\\model'
    import math
    directory = output_directory

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    epoch_counter = 1
    sess.run(model.init)
    graph_writter = tf.summary.FileWriter(directory, graph=tf.get_default_graph())

    height, width = feature_array.shape
    train_height = math.ceil(height * train_test_split)
    train_feature_set = feature_array[:train_height, :]
    print(train_feature_set.shape)
    train_gt_set = gt_array[:train_height, :]
    print(train_gt_set.shape)

    val_feature_set = feature_array[train_height:height, :]
    print(val_feature_set.shape)
    val_gt_set = gt_array[train_height:height, :]
    print(val_gt_set.shape)



    while epoch_counter <= epochs:
        saver = tf.train.Saver(max_to_keep=500)
        print(str(datetime.datetime.now()))
        print('training for epoch ', epoch_counter)
        for counter, feature in enumerate(train_feature_set):
            feature = feature.reshape(1,300)
            gt =train_gt_set[1,:]
            gt = gt.reshape(1,4)
            #print('gt shape ', gt.shape)
            sess.run(model.optimize, feed_dict={model.dataset_holder: feature, model.gt_holder : gt})
        checkpoint_number = epoch_counter
        epoch_counter = epoch_counter + 1
        save_checkpoint(sess, saver, directory, checkpoint_number)

        total_loss = 0
        total_counter = 0
        for counter, feature in enumerate(val_feature_set):
            feature = feature.reshape(1, 300)
            gt = val_gt_set[1, :]
            gt = gt.reshape(1, 4)
            predicted = sess.run(model.output_layer, feed_dict={model.dataset_holder: feature, model.gt_holder: gt})
            #print('predicted value ',predicted)
            loss = sess.run(model.loss, feed_dict={model.dataset_holder: feature, model.gt_holder: gt})
            total_loss += loss
            #print('loss is', loss)
            total_counter  = counter + 1

        print('average loss for epoch ', epoch_counter, ' is ', total_loss/total_counter)

        dummy_feature = numpy.ones((1,300))
        dummy_gt = numpy.ones((1,4))
        summary_str = sess.run(model.merge, feed_dict={model.dataset_holder: dummy_feature , model.gt_holder: dummy_gt})

        graph_writter.add_summary(summary_str, epoch_counter)
        graph_writter.flush()

    graph_writter.close()
    sess.close()



