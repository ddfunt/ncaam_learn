from keras_learn import create_model, load_data, features_labels, get_test_set, compile_model


data = load_data()
train_features, train_labels = features_labels(data)
test_features, test_labels, train_features, train_labels = get_test_set(train_features, train_labels, n=1000)

model = create_model()
compile_model(model)
checkpoint_path = "training_1/cp.ckpt"

loss, acc = model.evaluate(test_features, test_labels)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))

model.load_weights(checkpoint_path)
loss, acc = model.evaluate(test_features, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
print(loss)

import  numpy as np
predictions = model.predict(np.array(test_features))
print(predictions)
print(len(predictions))
print(predictions[0])
print(np.argmax(predictions[0]))
print(test_labels[np.argmax(predictions[0])])
print(predictions[1])
print(np.argmax(predictions[1]))
print(test_labels[np.argmax(predictions[1])])
print(predictions[2])
print(np.argmax(predictions[2]))
print(test_labels[np.argmax(predictions[2])])
print(predictions[3])
print(np.argmax(predictions[3]))
print(test_labels[np.argmax(predictions[3])])