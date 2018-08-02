# Hint: you should refer to the API in https://github.com/tensorflow/tensorflow/tree/r1.0/tensorflow/contrib
# Use print(xxx) instead of print xxx
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import shutil
import os
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '13'


# Global config, please don't modify
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.20
sess = tf.Session(config=config)
model_dir = r'../model'

# Dataset location
DEPRESSION_DATASET = '../data/data.csv'
DEPRESSION_TRAIN = '../data/training_data.csv'
DEPRESSION_TEST = '../data/testing_data.csv'

# Delete the exist model directory
if os.path.exists(model_dir):
    shutil.rmtree(model_dir)



# TODO: 1. Split data (5%)

# Split data: split file DEPRESSION_DATASET into DEPRESSION_TRAIN and DEPRESSION_TEST with a ratio about 0.6:0.4.
# Hint: first read DEPRESSION_DATASET, then output each line to DEPRESSION_TRAIN or DEPRESSION_TEST by use
# random.random() to get a random real number between 0 and 1.


# Reference https://docs.python.org/2/library/random.html
#https://stackoverflow.com/questions/17412439/how-to-split-data-into-trainset-and-testset-randomly
#https://cs230-stanford.github.io/train-dev-test-split.html

datafile = open(DEPRESSION_DATASET)
train_data = open(DEPRESSION_TRAIN, 'w')
test_data = open(DEPRESSION_TEST, 'w')

'''
#Method 1
with open(datafile, "rb") as f:
    data = f.read().split('\n')

random.shuffle(data)

train_data = data[:60]
test_data = data[60:]

'''

#Method 2
'''
for raw in datafile.readlines():
    datafile.sort()  # make sure that the filenames have a fixed order before shuffling
    random.seed(230)
    random.shuffle(datafile) # shuffles the ordering of filenames (deterministic given the chosen seed)
    split_1 = int(0.6 * len(filenames))
    split_2 = int(0.4 * len(filenames))
    train_filenames = datafile[:split_1]
    train_data.write(train_filenames)
    dev_filenames = datafile[split_1:split_2]
    test_filenames = datafile[split_2:]
    test_data.write(test_filenames)



'''

#0.6 Training Data
#0.4 Testing Data
#Method 3
train_ratio = 0.6
for raw in datafile.readlines():
    if random.random() < train_ratio:
        train_data.write(raw)

    else:
        test_data.write(raw)
#print ("Training amount",train_data)
#print ("Testing amount",test_data)
datafile.close()
train_data.close()
test_data.close()

# Reference https://www.tensorflow.org/versions/r1.1/get_started/tflear

# TODO: 2. Load data (5%)

training_set = tf.contrib.learn.datasets.base.load_csv_without_header(
    filename=DEPRESSION_TRAIN,
    target_dtype=np.int32,
    features_dtype=np.float32)

test_set = tf.contrib.learn.datasets.base.load_csv_without_header(
    filename=DEPRESSION_TEST,
    target_dtype=np.int32,
    features_dtype=np.float32)

features_train = tf.constant(training_set.data)
features_test = tf.constant(test_set.data)
labels_train = tf.constant(training_set.target)
labels_test = tf.constant(test_set.target)

# TODO: 3. Normalize data (15%)

normalize = tf.concat(axis=0, values=[features_train, features_test])
# or
'''
Reference : https://www.tensorflow.org/api_docs/python/tf/nn/l2_normalize
tf.nn.l2_normalize(
    x,
    axis=None,
    epsilon=1e-12,
    name=None,
    dim=None
)
'''
normalize = tf.nn.l2_normalize(x=normalize, dim=0)
# slice from 0,0 to training  data and then from trainning  data to test data



features_train = tf.slice(normalize, [0, 0], [len(training_set.data), -1])
features_test = tf.slice(normalize, [len(training_set.data), 0], [len(test_set.data), -1])


# Hint:
# we must normalize all the data at the same time, so we should combine the training set and testing set
# firstly, and split them apart after normalization. After this step, your features_train and features_test will be
# new feature tensors.
# Some functions you may need: tf.nn.l2_normalize, tf.concat, tf.slice

# TODO: 4. Build linear classifier with `tf.contrib.learn` (5%)
#dim = datafile.data.size[1]
#print (dim)

# we can get this from the csv file
dim = 112 #How many dimensions our feature have
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=dim)]

# You should fill in the argument of LinearClassifier

###################################################Linear_Classifier#######################

#classifier = tf.contrib.learn.LinearClassifier(feature_columns=feature_columns,model_dir=model_dir, n_classes=2, optimizer=tf.train.AdamOptimizer(0.01))
###################################################Linear_classier######################################
# TODO: 5. Build DNN classifier with `tf.contrib.learn` (5%)

# You should fill in the argument of DNNClassifier
###########################DNN#########################################
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[64,32,16,8,64],
                                            n_classes=2,
                                            model_dir=model_dir)
###############################END_DNN#########################################################
#https://www.tensorflow.org/api_docs/python/tf/contrib/learn/LinearClassifier
#linear_classifier = tf.contrib.learn.LinearClassifier(feature_columns)
# Define the training inputs
def get_train_inputs():
    x = tf.constant(features_train.eval(session=sess))
    y = tf.constant(labels_train.eval(session=sess))

    return x, y

# Define the test inputs
def get_test_inputs():
    x = tf.constant(features_test.eval(session=sess))
    y = tf.constant(labels_test.eval(session=sess))

    return x, y

# TODO: 6. Fit model. (5%)


classifier.fit(input_fn=get_train_inputs, steps=400)



validation_metrics = {
    "true_negatives":
        tf.contrib.learn.MetricSpec(
            metric_fn=tf.contrib.metrics.streaming_true_negatives,
            prediction_key=tf.contrib.learn.PredictionKey.CLASSES
        ),
    "true_positives":
        tf.contrib.learn.MetricSpec(
            metric_fn=tf.contrib.metrics.streaming_true_positives,
            prediction_key=tf.contrib.learn.PredictionKey.CLASSES
        ),
    "false_negatives":
        tf.contrib.learn.MetricSpec(
            metric_fn=tf.contrib.metrics.streaming_false_negatives,
            prediction_key=tf.contrib.learn.PredictionKey.CLASSES
        ),
    "false_positives":
        tf.contrib.learn.MetricSpec(
            metric_fn=tf.contrib.metrics.streaming_false_positives,
            prediction_key=tf.contrib.learn.PredictionKey.CLASSES
        ),
}

# TODO: 7. Make Evaluation (10%)

# evaluate the model and get TN, FN, TP, FP
result = classifier.evaluate(input_fn=get_test_inputs,
                             steps=1
                             , metrics=validation_metrics)

TN = result["true_negatives"]
FN = result["false_negatives"]
TP = result["true_positives"]
FP = result["false_positives"]

# You should evaluate your model in following metrics and print the result:
# Accuracy

# Precision in macro-average

# Recall in macro-average


acc = (TN + TP) / (TN + FN + TP + FP)
print ("Accuracy",acc)


pr_pos = TP / (TP + FP)
pr_neg = TN / (TN + FN)
pre_mac = (pr_pos + pr_neg)
pre_mac = pre_mac/2
print ("Precioson in macro-average",pre_mac)


re_pos = TP / (TP + FN)
re_neg = TN / (TN + FP)
re_mac = (re_pos + re_neg)
re_mac = re_mac/2
print ("Recall in macro-average",re_mac)


f1_score_pos = 2 * pr_pos * re_pos / (pr_pos + re_pos)
f1_score_neg = 2 * pr_neg * re_neg / (pr_neg + re_neg)
f1_score_macro = (f1_score_neg + f1_score_pos) / 2
print ("F1-score in macro-average",f1_score_macro)
