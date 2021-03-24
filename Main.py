from collections import deque
from datetime import time
from pyOpenBCI import OpenBCICyton
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import pandas as pd
from pylsl import StreamInlet, resolve_stream
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization
import os
import random


#printing raw data

def print_raw(sample):
    print(sample.channels_data)
board = OpenBCICyton(daisy=True) #the daisy is the extra 8 channels
board.start_stream(print_raw) #prints out the raw data of the stream
#each separate channel is going to be 1 separate electrode on the device connected to your head, but i don't have one (✖╭╮✖)


style.use("ggplot")

fps_counter = deque(maxlen=100)
FPS = 105 #105 FPS/Hz
HM_SECOND_SLICE = 100 #how many seconds we want to slice

data = np.load("sequence-45000.npy") #reads from this file with numpy, this is an array of arrays (not too sure if that's called a jagged array?)

for i in range (FPS*HM_SECOND_SLICE, len(data)):
    data2 = data[i-FPS*HM_SECOND_SLICE: i]
    c8 = data2[:, 8] #channel 9 on the headset, because it is indexed starting from 0
               #^^^ we want the 8th index of all of the arrays we're referencing
    GRAPH = c8
    print(c8)
    time.sleep(1/FPS)

    plt.plot(c8)
    plt.show()

    #a moving average using pandas (my own code, not from a tutorial)
    c8_series = pd.Series(c8)
    c8w = c8_series.rolling(HM_SECOND_SLICE)
    moving_average = c8w.mean()

    plt.plot(moving_average)
    break

last_print  = time.time()
fps_counter = deque(maxlen=150)

#resolve an EEG stream
streams = resolve_stream('type', 'EEG')
#create an inlet to read from the EEG stream
inlet = StreamInlet(streams[0])

channel_data = {}

for i in range(10):
    for i in range[16]: #recording from each of the 16 channels, which give us 125 data points
        sample, timestamp, = inlet.pull_sample()    #creating channel data to display, 10 iterations
        if i not in channel_data:                   #
            channel_data[i] = sample                #
        else:                                       #
            channel_data[i].append(sample)          #
    fps_counter.append(time.time() - last_print)
    last_print = time.time()
    raw_hz = 1/(sum(fps_counter)/len(fps_counter))
    print(raw_hz)

for chan in channel_data: #for channels in the channel data
    plt.plot(channel_data[chan][:125]) #plot the channel data of each channel for 125 data points
plt.show()

#building an AI to predict thoughts depending on the brain signals it is given

ACTIONS = ["left", "right"]
reshape = (-1, 16, 60)

def create_data(starting_dir="data"):
    training_data = {}
    for action in ACTIONS:
        if action not in training_data:
            training_data[action] = []

        data_dir = os.path.join(starting_dir,action)
        for item in os.listdir(data_dir):
            data = np.load(os.path.join(data_dir, item))
            for item in data:
                training_data[action].append(item)

    lengths = [len(training_data[action]) for action in ACTIONS]
    print(lengths)

    for action in ACTIONS:
        np.random.shuffle(training_data[action])  # note that regular shuffle is GOOF af
        training_data[action] = training_data[action][:min(lengths)]

    lengths = [len(training_data[action]) for action in ACTIONS]
    print(lengths)
    combined_data = []
    for action in ACTIONS:
        for data in training_data[action]:

            if action == "left":
                combined_data.append([data, [1, 0, 0]])

            elif action == "right":
                combined_data.append([data, [0, 0, 1]])


    np.random.shuffle(combined_data)
    print("length:",len(combined_data))
    return combined_data


traindata = create_data(starting_dir="data")
train_x = []
train_y = []
for X, y in traindata:
    train_x.append(X)
    train_y.append(y)

testdata = create_data(starting_dir="validation_data")
test_x = []
test_y = []
for x, y in testdata:
    test_x.append(X)
    test_y.append(y)

print(len(train_x))
print(len(test_x))


print(np.array(train_x).shape)
train_X = np.array(train_x).reshape(reshape)
test_X = np.array(test_x).reshape(reshape)

train_y = np.array(train_y)
test_y = np.array(test_y)

model = Sequential()

model.add(Conv1D(64, (3), input_shape=train_X.shape[1:]))
model.add(Activation('relu'))

model.add(Conv1D(64, (2)))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=(2)))

model.add(Conv1D(64, (2)))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=(2)))

model.add(Flatten())

model.add(Dense(512))

model.add(Dense(3))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

epochs = 10
batch_size = 32
for epoch in range(epochs):
    model.fit(train_X, train_y, batch_size=batch_size, epochs=1, validation_data=(test_X, test_y))
    score = model.evaluate(test_X, test_y, batch_size=batch_size)
