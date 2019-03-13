import tensorflow as tf


def load_csv( filename='game_data.csv'):
    with open(filename, 'r') as f:
        headers = f.readline().replace('/', '_').split(',')
        data = f.readlines()
    return headers, data

class DataClass:

    def __init__(self, headers, data_row):
        data_row = data_row.split(',')
        for name, d in zip(headers, data_row):
            try:
                setattr(self, name, float(d))
            except:
                setattr(self, name, d.strip())


"""
date, rank, opponent, outcome, team_score, 
opp_score, home/away, rank,name,conf,
win_loss,adjEM,adjO,adjD,adjT,
luck,adjEM,oppO,oppD,noncon_adjEM, 
opp_rank, opp_name, opp_conf, opp_win_loss, opp_adjEM, 
opp_adjO, opp_adjD, opp_adjT, opp_luck, opp_adjEM, 
opp_oppO, opp_oppD, opp_noncon_adjEM

"""


#tf.feature_column.numeric_column(key='rank') #1
tf.feature_column.numeric_column(key='outcome') #3
tf.feature_column.numeric_column(key='team_score') #4
tf.feature_column.numeric_column(key='opp_score') #5
tf.feature_column.numeric_column(key='home_away') #6
tf.feature_column.numeric_column(key='conf') #9
tf.feature_column.numeric_column(key='win_loss') #10
tf.feature_column.numeric_column(key='adjEM')
tf.feature_column.numeric_column(key='adjO')
tf.feature_column.numeric_column(key='adjD')
tf.feature_column.numeric_column(key='adjT')
tf.feature_column.numeric_column(key='luck')
tf.feature_column.numeric_column(key='adjEM')
tf.feature_column.numeric_column(key='oppO')
tf.feature_column.numeric_column(key='oppD')
tf.feature_column.numeric_column(key='noncon_adjEM')
tf.feature_column.numeric_column(key='opp_rank')
tf.feature_column.numeric_column(key='opp_name')
tf.feature_column.numeric_column(key='opp_conf')
tf.feature_column.numeric_column(key='opp_win_loss')
tf.feature_column.numeric_column(key='opp_adjEM')
tf.feature_column.numeric_column(key='opp_adjO')
tf.feature_column.numeric_column(key='opp_adjD')
tf.feature_column.numeric_column(key='opp_adjT')
tf.feature_column.numeric_column(key='opp_luck')
tf.feature_column.numeric_column(key='opp_adjEM')
tf.feature_column.numeric_column(key='opp_oppO')
tf.feature_column.numeric_column(key='opp_oppD')
tf.feature_column.numeric_column(key='opp_noncon_adjEM')




"""
model = tf.estimator.DNNClassifier(
  model_dir='model/',
  hidden_units=[10],
  feature_columns=feature_columns,
  n_classes=3,
  label_vocabulary=['H', 'D', 'A'],
  optimizer=tf.train.ProximalAdagradOptimizer(
    learning_rate=0.1,
    l1_regularization_strength=0.001
  ))"""


if __name__ == '__main__':
    data = []
    h, d = load_csv()
    for row in d:
        data.append(DataClass(h,row))
    print(data[0].name)