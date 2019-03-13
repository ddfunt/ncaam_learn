from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import numpy as np



def load_csv( year):
    filename = 'game_data_{}.csv'.format(year)
    with open(filename, 'r') as f:
        headers = f.readline().replace('/', '_').split(',')
        data = f.readlines()
    return headers, data

class DataClass:


    def __init__(self, headers, data_row):
        data_row = data_row.split(',')
        for name, d in zip(headers, data_row):
            try:
                setattr(self, name.strip(), float(d))
            except:
                setattr(self, name.strip(), d.strip())

    @property
    def win_loss(self):
        return self._win_loss

    @win_loss.setter
    def win_loss(self, val):
        x = val.split('-')
        self._win_loss =  float(x[0])-float(x[1])

    @property
    def opp_win_loss(self):
        return self._opp_win_loss

    @opp_win_loss.setter
    def opp_win_loss(self, val):
        x = val.split('-')
        self._opp_win_loss = float(x[0]) - float(x[1])

    @property
    def home_away(self):
        return self._home_away

    @win_loss.setter
    def home_away(self, val):
        if val.lower() == 'h':
            self._home_away = 2
        elif val.lower() == 'n':
            self._home_away = 1
        else:
            self._home_away = 0

    @property
    def outcome(self):
        return self._outcome

    @outcome.setter
    def outcome(self, value):
        if value.lower() == 'w':
            self._outcome = 1
        else:
            self._outcome = 0


def load_data():
    data = []
    for year in [2018, 2017, 2016, 2015, 2014]:
        h, d = load_csv(year)
        for row in d:
            data.append(DataClass(h,row))
    return data

def features_labels(data):
    feature_columns = [

            tf.feature_column.numeric_column(key='team_score'), #4
            tf.feature_column.numeric_column(key='opp_score'), #5
            tf.feature_column.numeric_column(key='home_away'), #6
            #tf.feature_column.numeric_column(key='conf'), #9
            tf.feature_column.numeric_column(key='win_loss'), #10
            tf.feature_column.numeric_column(key='adjEM'),
            tf.feature_column.numeric_column(key='adjO'),
            tf.feature_column.numeric_column(key='adjD'),
            tf.feature_column.numeric_column(key='adjT'),
            tf.feature_column.numeric_column(key='luck'),
            tf.feature_column.numeric_column(key='oppO'),
            tf.feature_column.numeric_column(key='oppD'),
            tf.feature_column.numeric_column(key='noncon_adjEM'),
            tf.feature_column.numeric_column(key='opp_rank'),
            #tf.feature_column.numeric_column(key='opp_name'),
           # tf.feature_column.numeric_column(key='opp_conf'),
            tf.feature_column.numeric_column(key='opp_win_loss'),
            tf.feature_column.numeric_column(key='opp_adjEM'),
            tf.feature_column.numeric_column(key='opp_adjO'),
            tf.feature_column.numeric_column(key='opp_adjD'),
            tf.feature_column.numeric_column(key='opp_adjT'),
            tf.feature_column.numeric_column(key='opp_luck'),
            tf.feature_column.numeric_column(key='opp_oppO'),
            tf.feature_column.numeric_column(key='opp_oppD'),
            tf.feature_column.numeric_column(key='opp_noncon_adjEM')
            ]


    train_features = {


        #'team_score': np.array([x.team_score for x in data]),
        #'rank': np.array([x.rank for x in data]),
        #'opp_score': np.array([x.opp_score for x in data]), #5
        'home_away': np.array([x.home_away for x in data]), #6
        #'conf': np.array([x.conf for x in data]), #9
        'win_loss': np.array([x.win_loss for x in data]), #10
        'adjEM': np.array([x.adjEM for x in data]),
        'adjO': np.array([x.adjO for x in data]),
        'adjD': np.array([x.adjD for x in data]),
        'adjT': np.array([x.adjT for x in data]),
        'luck': np.array([x.luck for x in data]),

        'oppO': np.array([x.oppO for x in data]),
        'oppD': np.array([x.oppD for x in data]),
        'noncon_adjEM': np.array([x.noncon_adjEM for x in data]),

        'opp_rank': np.array([x.opp_rank for x in data]),
        #'opp_name': np.array([x.opp_name for x in data]),
        #'opp_conf': np.array([x.opp_conf for x in data]),
        'opp_win_loss': np.array([x.opp_win_loss for x in data]),

        'opp_adjO': np.array([x.opp_adjO for x in data]),
        'opp_adjD': np.array([x.opp_adjD for x in data]),
        'opp_adjT': np.array([x.opp_adjT for x in data]),
        'opp_luck': np.array([x.opp_luck for x in data]),
        'opp_adjEM': np.array([x.opp_adjEM for x in data]),
        'opp_oppO': np.array([x.opp_oppO for x in data]),
        'opp_oppD': np.array([x.opp_oppD for x in data]),
        'opp_noncon_adjEM': np.array([x.opp_noncon_adjEM for x in data])
          #'home-goals': np.array([7, 3, 4]),
          #'home-opposition-goals': np.array([3, 8, 6]),
          ## ... for each feature
    }
    train_features = np.array([x for _, x in train_features.items()]).transpose().tolist()

    train_labels = [d.outcome for d in data]#)

    return train_features, train_labels


def create_model():
    model = keras.Sequential([
        #keras.layers.Flatten(input_shape=(28, 28)),
        #keras.layers.
        keras.layers.Dense(16, activation=tf.nn.relu),
        keras.layers.Dense(2, activation=tf.nn.softmax)
    ])
    return model

def compile_model(model):

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
import random

def get_test_set(train_features, train_labels, n=100):
    test_features = []
    test_labels = []
    for i in range(n):
        blah = random.randint(0, len(train_features))
        test_features.append(train_features.pop(blah))
        test_labels.append(train_labels.pop(blah))
    test_labels = np.array(test_labels)
    test_features = np.array(test_features)
    train_labels = np.array(train_labels)
    train_features = np.array(train_features)
    return test_features, test_labels, train_features, train_labels

import os

def checkpoint_callback():
    checkpoint_path = "training_1/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create checkpoint callback
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)
    return cp_callback


def train(model,train_features, train_labels,  callback=None, epochs=10):
    model.fit(train_features, train_labels, epochs=epochs, callbacks=[callback])


if __name__ == '__main__':
    data = load_data()
    train_features, train_labels = features_labels(data)
    model = create_model()
    test_features, test_labels, train_features, train_labels = get_test_set(train_features, train_labels)
    callback = checkpoint_callback()
    compile_model(model)
    train(model,train_features, train_labels,  callback=callback, epochs=10000)

    test_loss, test_acc = model.evaluate(test_features, test_labels)

    print('Test accuracy:', test_acc)