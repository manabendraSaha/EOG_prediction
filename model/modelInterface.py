import tensorflow as tf
import model
import os
import numpy


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




def train_encoder(feature_array, gt_array):
    directory = 'C:\\projects\\anirban\\output\\model'

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    epochs = 10
    epoch_counter = 1
    sess.run(model.init)
    graph_writter = tf.summary.FileWriter(directory, graph=tf.get_default_graph())


    while epoch_counter <= epochs:
        saver = tf.train.Saver(max_to_keep=500)
        print('training for epoch ', epoch_counter)
        for counter, feature in enumerate(feature_array):
            feature = feature.reshape(1,300)
            gt =gt_array[1,:]
            gt = gt.reshape(1,4)
            #print('gt shape ', gt.shape)
            sess.run(model.optimize, feed_dict={model.dataset_holder: feature, model.gt_holder : gt})
        checkpoint_number = epoch_counter
        epoch_counter = epoch_counter + 1
        save_checkpoint(sess, saver, directory, checkpoint_number)


        for counter, feature in enumerate(feature_array):
            feature = feature.reshape(1, 300)
            gt = gt_array[1, :]
            gt = gt.reshape(1, 4)
            predicted = sess.run(model.output_layer, feed_dict={model.dataset_holder: feature, model.gt_holder: gt})
            print('predicted value ',predicted)
            loss = sess.run(model.loss, feed_dict={model.dataset_holder: feature, model.gt_holder: gt})
            print('loss is', loss)
            break

        dummy_feature = numpy.ones((1,300))
        dummy_gt = numpy.ones((1,4))
        summary_str = sess.run(model.merge, feed_dict={model.dataset_holder: dummy_feature , model.gt_holder: dummy_gt})

        graph_writter.add_summary(summary_str, epoch_counter)
        graph_writter.flush()

    graph_writter.close()
    sess.close()



