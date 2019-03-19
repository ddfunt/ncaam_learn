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

    #data =

    filename = 'team_data_{}.csv'.format(year)
    with open(filename, 'r') as f:
        team_headers = f.readline().replace('/', '_').split(',')
        team_data = f.readlines()
    return headers, data, team_headers, team_data

def load_data(years =[2015]):
    data = []
    teams = {}
    games = []
    for year in years:#[2018, 2017, 2016, 2015, 2014]:
        gh, gd, h, d = load_csv(year)
        for team in d:
            t = Team(h, team, year)
            #if len(t.misc)
            teams[f'{t.name}_{year}'] = t
        for row in gd:
            #print(teams.keys())
            g = Game(teams, row, year)
            if not g.flag:
                games.append(g)
    return teams, games

class Team:

    def __init__(self, headers, data, year):
        self.year = year
        self.keys = headers
        data = data.split(',')
        #input()
        for d, header in zip(data, headers):
            if header.strip():
                if header == 'misc':
                    z = d.split('^')
                    # print(z)
                    try:
                        setattr(self, header, list(map(float, z)))
                    except Exception as e:
                        setattr(self, header, [])
                else:
                    try:
                        setattr(self, header.strip(), float(d))
                    except:
                        setattr(self, header.strip(), d.strip())
                #print(header.strip(), d)

    @property
    def win_loss(self):
        return self._win_loss

    @win_loss.setter
    def win_loss(self, val):
        x = val.split('-')
        self._win_loss = float(x[0]) - float(x[1])

    @property
    def list_data(self):
        #self.win_loss,
        data = [self.adjEM, self.adjD, self.adjT, self.luck,]
                 #self.oppO, self.oppD, self.noncon_adjEM]
        #data.extend(self.misc)
        data = self.misc
        return data

class Game:
    _game_state = None

    def __init__(self, team_data, game_state, year):
        self.year = year
        #print(game_state)
        self.game_state = game_state
        self.team_1 = team_data[f"{self.team_1_name}_{year}"]
        self.team_2 = team_data[f'{self.team_2_name}_{year}']
        if len(self.team_1.misc) < 10 or len(self.team_2.misc) < 10:
            self.flag = True
        else:
            self.flag = False


    @property
    def game_state(self):
        return self._game_state

    @game_state.setter
    def game_state(self, val):

        d = val.strip().split(',')
        self.team_2_name = d[0]
        outcome = d[1]
        if 'W' in outcome:
            self.outcome = 1
        else:
            self.outcome = 0
        home_away = d[2]
        if 'H' in home_away:
            self.home_away = 2
        elif 'N' in home_away:
            self.home_away = 1
        else:
            self.home_away = 0

        self.team_1_name = d[3]
        self._game_state = d


    @property
    def list_data(self):

        data = [self.home_away]
        data.extend(self.team_1.list_data)
        data.extend(self.team_2.list_data)
        return np.array(data)



def features_labels(games):

    train_features = [game.list_data for game in games]
    #print(train_features[0])

    train_labels = [d.outcome for d in games]#)

    return train_features, train_labels


def create_model():
    model = keras.Sequential([
        #keras.layers.Flatten(input_shape=(65, 1)),
        #keras.layers.
        keras.layers.Dense(60, activation=tf.nn.relu),
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
        #print((len(train_features)))
        blah = random.randint(0, len(train_features) -1)
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
    team_data, game_data = load_data([ 2016, 2017, 2018, 2019])

    train_features, train_labels = features_labels(game_data)

    model = create_model()
    test_features, test_labels, train_features, train_labels = get_test_set(train_features, train_labels, n=1000)

    #print(np.shape(test_features))

    callback = checkpoint_callback()
    compile_model(model)
    train(model, np.array(train_features), np.array(train_labels),  callback=callback, epochs=3000)


    test_loss, test_acc = model.evaluate(test_features, test_labels)

    print('Test accuracy:', test_acc)