"""
Gradient-based learning applied to document recognition
python 3.6
tensorflow 2.1.0
conda activate astroboi_tf_2

with CUDA
CUDA 10.1
cudnn 7.6.0
conda activate astroboi_cuda_2
"""
import os
import numpy as np
from time import time

from tensorflow.keras import utils
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard

training_data = np.loadtxt('../input/mnist_train.csv', delimiter=',', dtype=np.float32)
test_data = np.loadtxt('../input/mnist_test.csv', delimiter=',', dtype=np.float32)

print('training_data.shape :', training_data.shape, ', test_data.shape :', test_data.shape)

X_train = training_data[:, 1:]
y_train = training_data[:, [0]]
print('X_train.shape :', X_train.shape, ', y_train.shape :', y_train.shape)

X_test = test_data[:, 1:]
y_test = test_data[:, [0]]
print('X_test.shape :', X_test.shape, ', y_test.shape :', y_test.shape)

# padding MNIST image, MNIST image is 28*28 => input of LeNet-5 is 32*32
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_train = np.pad(X_train, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_test = np.pad(X_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')

X_train = X_train / 255
X_test = X_test / 255

y_train = utils.to_categorical(y_train)
y_test = utils.to_categorical(y_test)

print('X_train.shape :', X_train.shape, ', y_train.shape :', y_train.shape)
print('X_test.shape :', X_test.shape, ', y_test.shape :', y_test.shape)


###################### st LeNet-5 structure ##################################
# Instantiate an empty sequential model
model = Sequential()
# C1 Convolutional Layer
model.add(layers.Conv2D(filters=6, kernel_size=5, strides=1, activation='tanh', input_shape=(32, 32, 1), padding='same'))

# S2 Pooling Layer
model.add(layers.AveragePooling2D(pool_size=2, strides=2, padding='valid'))

# C3 Convolutional Layer
model.add(layers.Conv2D(filters=16, kernel_size=5, strides=1, activation='tanh', padding='valid'))
# S4 Pooling Layer
model.add(layers.AveragePooling2D(pool_size=2, strides=2, padding='valid'))

# C5 Convolutional Layer
model.add(layers.Conv2D(filters=120, kernel_size=5, strides=1, activation='tanh', padding='valid'))

# Flatten the CNN output to feed it with fully connected layers
model.add(layers.Flatten())

# FC6 Fully Connected Layer
model.add(layers.Dense(units=84, activation='tanh'))

# FC7 Output layer with softmax activation
model.add(layers.Dense(units=10, activation='softmax'))

# print the model summary
model.summary()
###################### en LeNet-5 structure ##################################

###################### st Set up the learning hyperparameters ################
def lr_schedule(epoch):
    # initiate the learning rate with value = 0.0005
    lr = 5e-4
    # lr = 0.0005 for the first two epochs, 0.0002 for the next three epochs,
    # 0.00005 for the next four, then 0.00001 thereafter.
    if epoch > 2:
        lr = 2e-4
    elif epoch > 5:
        lr = 5e-5
    elif epoch > 9:
        lr = 1e-5
    return lr

MODEL_SAVE_DIR = './model/'
os.makedirs(MODEL_SAVE_DIR + '/chck_pnt/', exist_ok=True)
model_path = MODEL_SAVE_DIR + '/chck_pnt/{epoch}-{val_loss:.2f}-{val_accuracy:.2}.h5'

early_stopping = EarlyStopping(monitor='val_loss', patience=5)

checkpointer = ModelCheckpoint(filepath=model_path
                               # , save_weights_only=True
                               , monitor='val_loss'
                               , verbose=1
                               # , mode='max'
                               , save_best_only=True
                               )
###################### en Set up the learning hyperparameters ################

###################### st learning 1 #########################################
model.compile(loss='categorical_crossentropy'
              , optimizer=optimizers.SGD(lr=lr_schedule(0))
              , metrics=['accuracy']
              )

hist = model.fit(X_train, y_train
                 , batch_size=32
                 , epochs=20
                 , validation_data=(X_test, y_test)
                 , callbacks=[early_stopping, checkpointer]
                 # , callbacks=[checkpointer]
                 , verbose=1
                 , shuffle=True
                 )
###################### en learning 1 #########################################

###################### st learning 2 #########################################
# model.compile(loss=losses.categorical_crossentropy
#               , optimizer=optimizers.SGD(lr=lr_schedule(0))
#               , metrics=['accuracy'])
#
# EPOCHS = 2
# BATCH_SIZE = 32
#
# train_generator = ImageDataGenerator().flow(X_train, y_train, batch_size=BATCH_SIZE)
# validation_generator = ImageDataGenerator().flow(X_test, y_test, batch_size=BATCH_SIZE)
#
# steps_per_epoch = X_train.shape[0]//BATCH_SIZE
# validation_steps = X_test.shape[0]//BATCH_SIZE
#
# # tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
# model.fit(train_generator
#           , steps_per_epoch=steps_per_epoch
#           , epochs=EPOCHS
#           , validation_data=validation_generator
#           , validation_steps=validation_steps
#           , shuffle=True
#           , callbacks=[early_stopping, checkpointer]
#           # , callbacks=[checkpointer]
#           )
###################### en learning 2 #########################################

###################### st save model #########################################
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
model.save(MODEL_SAVE_DIR, include_optimizer=True)
###################### en save model #########################################