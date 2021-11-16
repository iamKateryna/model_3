import tensorflow as tf
import tensorflow_addons as tfa
PATH = 'MyModel_tf'


class MyModel(object):
    def __init__(self):
        self.model = tf.keras.models.load_model(PATH, custom_objects={"F1Score": tfa.metrics.F1Score})

    def predict(self, X, *args):
        classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        Y = self.model.predict(X)
        Y = Y[0]*100
        probs = {classes[i]: Y[i] for i in range(len(classes))}

        return probs
