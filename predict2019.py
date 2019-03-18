from __future__ import absolute_import, division, print_function
from keras_learn import create_model, load_data, features_labels, get_test_set, compile_model


# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from keras_learn import load_data


def construct_game(team1, team2):
    d = [1]
    d.extend(team1.list_data)
    d.extend(team2.list_data)
    return np.array([np.array(d)])


model = create_model()
compile_model(model)
checkpoint_path = "training_1/cp.ckpt"

if __name__ == '__main__':
    teams, games = load_data([2019])
    #print(teams)
    team1 = teams['Virginia Tech_2019']
    team2 = teams['Saint Louis_2019'] #Maryland Eastern Shore
    game = construct_game(team1, team2)
    #print(game)

    #loss, acc = model.evaluate(game, np.array([1]))
    #print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))

    model.load_weights(checkpoint_path)
    #loss, acc = model.evaluate(game, np.array([1]))
    #print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
    #print(loss)


    game = construct_game(team1, team2)
    predictions = model.predict(game)[0]
    print(f'{team1.name} win:  {predictions[1]*100:.2f}%')
    print(f'{team1.name} loss: {predictions[0]*100:.2f}%')
    game2 = construct_game(team2, team1)
    predictions_invert = model.predict(game2)[0]
    print(f'{team2.name} win:  {predictions_invert[1]*100:.2f}%')
    print(f'{team2.name} loss: {predictions_invert[0]*100:.2f}%')
    print('')
    print(f'Unknown Diff: {(abs(predictions[1]-predictions_invert[0])*100):.4f}')