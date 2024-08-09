import time  # Importing the time module to work with time-related functions.

import tensorflow as tf  # Importing TensorFlow for building and training neural networks.
from tensorflow.keras.callbacks import Callback  # Importing the Callback class for custom training behaviors.

from models import \
    model  # Importing the model module, which likely contains the function to create the neural network model.


def model_wrapper(wts_path, train=False, to_save_as=False, model_path=None):
    # If a model path is provided, load the model from the specified file and return it.
    if model_path:
        return tf.keras.models.load_model(model_path)

    # Otherwise, get a new instance of the model using the get_model function from the model module.
    my_model = model.get_model()

    # If a weights path is provided, load the weights into the model.
    if wts_path:
        my_model.load_weights(wts_path)

    # If the train flag is set to True, proceed with training the model.
    if train:
        # Define a custom callback to stop training once the desired accuracy is achieved.
        class myCallback(Callback):
            def on_epoch_end(self, epoch, logs={}):
                # Stop training if both training and validation accuracy exceed 95%.
                if logs.get('accuracy') > 0.95 and logs.get('val_accuracy') > 0.95:
                    print('Stopping training')
                    my_model.stop_training = True

        # Instantiate the callback.
        callbacks = myCallback()

        # Load the MNIST dataset (a dataset of handwritten digits).
        mnist = tf.keras.datasets.mnist

        # Split the dataset into training and testing sets.
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # Normalize the data to have values between 0 and 1 by dividing by 255.
        x_train = x_train / 255.0
        x_test = x_test / 255.0

        # Train the model on the training data for 10 epochs, using the custom callback.
        my_model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])

        # Evaluate the model on the test data and print the results.
        print(my_model.evaluate(x_test, y_test))

        # Save the model's weights. If a weights path is provided, append a timestamp to the filename.
        if wts_path:
            my_model.save_weights('{}-{}'.format(wts_path, round(time.time())))
        else:
            # Save the model's weights to the specified file if no path is provided.
            my_model.save_weights(to_save_as)

    # Return the trained or loaded model.
    return my_model
