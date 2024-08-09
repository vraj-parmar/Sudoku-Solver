import tensorflow as tf

def get_model():
    # Define a Sequential model, which is a linear stack of layers.
    model = tf.keras.Sequential([
        # Input layer that expects images of shape (32, 32, 1).
        tf.keras.layers.InputLayer(input_shape=(32, 32, 1)),

        # First convolutional layer with 64 filters, 2x2 kernel size, ReLU activation, and same padding.
        tf.keras.layers.Conv2D(64, (2, 2), activation="relu", padding="same"),
        # Max pooling layer with a 2x2 pool size to reduce the spatial dimensions by half.
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Second convolutional layer with 128 filters, 2x2 kernel size, ReLU activation, and same padding.
        tf.keras.layers.Conv2D(128, (2, 2), activation="relu", padding="same"),
        # Max pooling layer with a 2x2 pool size.
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Third convolutional layer with 256 filters, 2x2 kernel size, ReLU activation, and same padding.
        tf.keras.layers.Conv2D(256, (2, 2), activation="relu", padding="same"),
        # Max pooling layer with a 2x2 pool size.
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Flatten layer to convert the 3D outputs from the previous layers into 1D feature vectors.
        tf.keras.layers.Flatten(),

        # Fully connected layer with 256 units and ReLU activation.
        tf.keras.layers.Dense(256, activation='relu'),
        # Fully connected layer with 128 units and ReLU activation.
        tf.keras.layers.Dense(128, activation='relu'),
        # Fully connected layer with 64 units and ReLU activation.
        tf.keras.layers.Dense(64, activation='relu'),

        # Dropout layer with a rate of 0.2 to prevent overfitting by randomly setting 20% of inputs to 0.
        tf.keras.layers.Dropout(0.2),

        # Output layer with 9 units (for 9 classes) and softmax activation to generate probability distributions.
        tf.keras.layers.Dense(9, activation="softmax")
    ])

    # Compile the model with the Adam optimizer, categorical crossentropy loss, and accuracy as a metric.
    model.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model
