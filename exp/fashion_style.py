# Use print(xxx) instead of print xxx
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from six.moves import range
import os


# Log level setting. (No need to modify.)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.ERROR)

# GPU memory configuration. (Do not modify.)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.2


# Path of csv data.
FASHION_TRAIN = '../data/fashion_train.csv'
FASHION_TEST = '../data/fashion_test.csv'

# Dimension of features and labels.
NUM_FEATURES = 137
NUM_LABELS = 2

# Load data from csv files.
fashion_train = np.genfromtxt(FASHION_TRAIN, delimiter=',')
fashion_test = np.genfromtxt(FASHION_TEST, delimiter=',')
print('Original data shape:', fashion_train.shape, fashion_test.shape)

# Split into feature and label matrix.
train_features, train_labels = fashion_train[:, :NUM_FEATURES], fashion_train[:, NUM_FEATURES:]
test_features, test_labels = fashion_test[:, :NUM_FEATURES], fashion_test[:, NUM_FEATURES:]
print('Training set size:', train_features.shape, train_labels.shape)
print('Test set size:', test_features.shape, test_labels.shape)

# Combine train_features and test_features into ae_features.
ae_features = np.concatenate((train_features, test_features))
print('AE set size:', ae_features.shape)


# Training parameters to be adjusted.
batch_size = 64
learning_rate = 0.01
num_steps = 10001

# Hidden layer size.
n_hidden_1 = 256
n_hidden_2 = 128


# Structure of autoencoder.
graph = tf.Graph()
with graph.as_default():

    # Input a batch of training data.
    tf_train_features = tf.placeholder(tf.float32, shape=(batch_size, NUM_FEATURES))

    # Weights and biases of encoder's first layer.
    encoder_w1 = tf.Variable(tf.truncated_normal([NUM_FEATURES, n_hidden_1]))
    encoder_b1 = tf.Variable(tf.zeros([n_hidden_1]))

    # TODO : 1. Create weights and biases of encoder's second layer. (5%)
    # Hint : use n_hidden_2
    encoder_w2 = tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2]))
    encoder_b2 = tf.Variable(tf.zeros([n_hidden_2]))

    # TODO : 2. Create weights and biases of encoder's *second* layer. (5%)
    # Hint : pay attention to the symmetry between layers
    
    decoder_w2 = tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_1]))
    decoder_b2 = tf.Variable(tf.zeros([n_hidden_1]))
    
    # Weights and biases of decoder's *first* layer.
    decoder_w1 = tf.Variable(tf.truncated_normal([n_hidden_1, NUM_FEATURES]))
    decoder_b1 = tf.Variable(tf.zeros([NUM_FEATURES]))

    # Training computation.
    encoder_l1 = tf.sigmoid(tf.matmul(tf_train_features, encoder_w1) + encoder_b1)
    # TODO : 3. Write the computation of encoder's second layer and decoder's *second* layer. (5%)
    # Hint : similar to encoder_l1 and decoder_l1
    
    encoder_l2 = tf.sigmoid(tf.matmul(encoder_l1, encoder_w2) + encoder_b2)
    decoder_l2 = tf.sigmoid(tf.matmul(encoder_l2, decoder_w2) + decoder_b2)    
    decoder_l1 = tf.sigmoid(tf.matmul(decoder_l2, decoder_w1) + decoder_b1)

    # TODO : 4. Define the loss function. (5%)
    # Hint : use tf.losses.mean_squared_error()
    loss = tf.losses.mean_squared_error(decoder_l1, tf_train_features)

    # TODO : 5. Define a gradient descent optimizer. (5%)
    # Hint : user tf.train.GradientDescentOptimizer(...).minimize(...)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)


# Training process.
with tf.Session(graph=graph, config=config) as session:
    # Initialize the variables.
    tf.global_variables_initializer().run()
    print('Initialized')

    # Autoencoder training process.
    for step in range(num_steps):
        # Pick an offset within the training data, which has been randomized.
        offset = (step * batch_size) % (ae_features.shape[0] - batch_size)

        # Generate a minibatch.
        batch_data = ae_features[offset:(offset + batch_size), :]

        # Prepare a dictionary telling the session where to feed the minibatch.
        feed_dict = {tf_train_features : batch_data}

        # Run the session and get the loss.
        _, l = session.run([optimizer, loss], feed_dict=feed_dict)

        # Print the loss every 500 steps.
        if step % 500 == 0:
            print('Minibatch loss at step %d: %.4f' % (step, l))

    # TODO : 6. Change train_features and test_features from numpy arrays to tensorflow constants. (5%)
    # Hint : use tf.constant(...); pay attention to "dtype" and "shape" parameters
    train_features = tf.constant(train_features, dtype=tf.float32, shape=train_features.shape)
    test_features = tf.constant(test_features, dtype=tf.float32, shape=test_features.shape)
    # TODO : 7. Calculate the middle layer representation of train/test features. (5%)
    # Hint : use tf.sigmoid(...) and tf.matmul(...); use encoder layers' weights/biases; add ".eval()" at the end of the expressions
    train_features_new = tf.sigmoid(tf.matmul(tf.sigmoid(tf.matmul(train_features, encoder_w1) + encoder_b1), encoder_w2) + encoder_b2).eval()
    test_features_new = tf.sigmoid(tf.matmul(tf.sigmoid(tf.matmul(test_features, encoder_w1) + encoder_b1), encoder_w2) + encoder_b2).eval()
 
    print('Middle layer representation size: ', train_features_new.shape, test_features_new.shape)

    # Input function for DNN regressor.
    def input_fn(features, label):
        feature_cols = {str(k): tf.constant(features[:, k]) for k in range(n_hidden_2)}
        label = tf.constant(label, dtype=tf.float32, shape=label.shape)
        return feature_cols, label

    total_loss = 0.0
    # Train regression model on 2 dimensions separately. (x- and y-axis on the Fashion Semantic Space)
    for index in [0, 1]:
        # Feature columns for DNN regressor.
        feature_cols = [tf.contrib.layers.real_valued_column(str(k)) for k in range(n_hidden_2)]
        regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols, hidden_units=[32, 16])
        # TODO : 8. Define DNN regressor. (5%)
        # Hint : use feature_cols; define hidden layers' size
        
        regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols, hidden_units=[32, 16])

        # TODO : 9. Train the regressor. (5%)
        # Hint : feed train features and label into input_fn(...)

        regressor.fit(input_fn=lambda: input_fn(train_features_new, train_labels[:, index]), steps=100)

        # TODO : 10. Evaluate the loss on test set. (5%)
        # Hint : feed test features and label into input_fn(...)
        current_loss = regressor.evaluate(input_fn=lambda: input_fn(test_features_new, test_labels[:, index]), steps=1)['loss']
       

        print('Index %d MSE loss: %.4f' % (index, current_loss))
        total_loss += current_loss

    # Print the sum of 2 dimensions' loss.
    print('Total MSE loss: %.4f' % total_loss)

