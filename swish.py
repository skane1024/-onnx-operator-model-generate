import tensorflow as tf
from tensorflow.keras.layers import Activation
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Define the Swish activation function
def swish(x):
    return tf.keras.backend.sigmoid(x) * x

# Register the Swish function as a custom activation
get_custom_objects().update({'swish': Activation(swish)})

# Create a model
model = Sequential()
model.add(Dense(64, input_dim=10))
model.add(Activation('swish'))
model.add(Dense(32))
model.add(Activation('swish'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Save the model as an H5 file
model.save('swish_model.h5')
