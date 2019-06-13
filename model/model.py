import tensorflow as tf

#activation = tf.nn.relu
activation = tf.nn.leaky_relu
initializer_kernel = tf.contrib.layers.xavier_initializer()
initializer_bias = tf.random_uniform_initializer()
dataset_holder = tf.placeholder(name='input_data', shape=[None, 300], dtype=tf.float32)
gt_holder = tf.placeholder(name='gt_data', shape=[None, 4], dtype=tf.float32)

with tf.name_scope('layer_1'):
    layer_1 =tf.layers.dense(inputs=dataset_holder, units=20, activation=activation,
                             trainable=True, use_bias=True, name='layer_1',
                             kernel_initializer=initializer_kernel, bias_initializer=initializer_bias)
    #dropout_layer_1 = tf.nn.dropout(layer_1, keep_prob = drop_out_keep_prob, name='droput_layer_1' )
    #output_1 = tf.cond(is_train, lambda : dropout_layer_1, lambda : layer_1 )
    tf.summary.histogram(name='name', values=layer_1)

with tf.name_scope('output_layer'):
    output_layer = tf.layers.dense(inputs=layer_1, units = 4, activation=activation, trainable=True,
                                   name='output_layer')
    tf.summary.histogram(name='output_layer', values=output_layer)

#loss = tf.losses.absolute_difference(labels=gt_holder, predictions=output_layer)
loss = tf.losses.mean_squared_error(labels=gt_holder, predictions=output_layer)


cost_function = tf.reduce_sum(loss)
#cost_function = tf.reduce_all(loss)
optimize = tf.train.RMSPropOptimizer(learning_rate= 0.001).minimize(cost_function)
#optimize = tf.train.AdamOptimizer(learning_rate= 0.001).minimize(cost_function)


merge = tf.summary.merge_all()




init = tf.global_variables_initializer()




