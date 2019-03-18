from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from keras_learn import DataClass


def load_csv():
    filename = 'team_data.csv'
    with open(filename, 'r') as f:
        headers = f.readline().replace('/', '_').split(',')
        d = f.readlines()

    data = {}
    for row in d:
        #print(row)
        dc = DataClass(headers, row)
        data[dc.name] = dc
    return data

def features_labels(team):

    train_features = [


        #'team_score': np.array([x.team_score for x in data]),
        #'rank': np.array([x.rank for x in data]),
        #'opp_score': np.array([x.opp_score for x in data]), #5
         #team.home_away, #6
        #'conf': np.array([x.conf for x in data]), #9
         team.win_loss,  #10
         team.adjEM,
         team.adjO,
         team.adjD,
         team.adjT,
         team.luck,

         team.oppO,
         team.oppD,
         team.noncon_adjEM,

         #np.array([x.opp_rank for x in data]),
        #'opp_name': np.array([x.opp_name for x in data]),onf': np.array([x.opp_conf for x in data]),
         #np.array([x.opp_win_loss for x in data]),

         #team.opp_adjO,
         #team.opp_adjD,
         #team.opp_adjT,
         #team.opp_luck,
         #team.opp_adjEM,
         #team.opp_oppO,
         #team.opp_oppD,
         #team.opp_noncon_adjEM

    ]

    for x in team.misc:
        train_features.append(x)

    return train_features

def construct_game(team1, team2):
    pass

if __name__ == '__main__':
    data = load_csv()
    team1 = features_labels(data['Virginia'])
    team2 = features_labels(data['Florida St.'])
    print(team1)
    print(len(team1))

