import numpy as np
import h5py
import scipy.io
np.random.seed(1337) # for reproducibility

from keras.preprocessing import sequence
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.regularizers import l2, activity_l1
from keras.constraints import maxnorm
from keras.layers.recurrent import LSTM, GRU
from keras.callbacks import ModelCheckpoint, EarlyStopping
from seya.layers.recurrent import Bidirectional
from keras.utils.layer_utils import print_layer_shapes


print 'loading data'

trainmat = h5py.File('data/train.mat')
validmat = scipy.io.loadmat('data/valid.mat')
testmat = scipy.io.loadmat('data/test.mat')

X_train = np.transpose(np.array(trainmat['trainxdata']),axes=(2,0,1))
y_train = np.array(trainmat['traindata']).T

# Creates an LSTM network. For each input tensor into an LSTM unit
# that tensor would have 320 values to process
# and the output of that LSTM unit would be a tensor with 320 values
# But what dictates how many unrolled LSTM units there will be, or really how
# much time series input data there will be? (maybe when you model.add the lstm layer, keras will figure this out?)
# By saying return_sequences=True, you say that we want to save the outputs
# of all LSTM inputs in the time series, not just the last output
# So for every input in the time series data, there will be an output
# that is the generated hidden state from that input after running it through all the gates
# in the lstm. return sequences essentially asks if we want to return the hidden state
# from each individual input or just the last of these states.
forward_lstm = LSTM(input_dim=320, output_dim=320, return_sequences=True)
# Same LSTM unit as forward_lstm
backward_lstm = LSTM(input_dim=320, output_dim=320, return_sequences=True)
# Creates a Bidirectional layer with the forward layer(going in one direction on the input)
# being the forward_lstm and the backward layer being the backward LSTM
# we also specificfy that in this layer we want the output after each input in the time series data
# not just the last output, by doing return_sequences = true
# How is each output decided? -> how are the outputs of forward layer and backward layer merged?
# Answer: because they did not specify merge_mode, the outputs of the forward and backward
# layers are not combined but rather both the outputs are returned by this layer
brnn = Bidirectional(forward=forward_lstm, backward=backward_lstm, return_sequences=True)

print 'building model'

# We declare our model to be a sequential model, which means each layer
# comes one after the other and that for every layer there is exackly one input tensor
# and one output tensor.
# Eventually when we have a network where input goes to both a R-loop folding net
# and to a DanQ/BPNet and then merge to an output, our model would not be Sequential
# We would probably have to use keras Functional API: https://machinelearningmastery.com/keras-functional-api-deep-learning/
model = Sequential()
# We add a one dimensional Convolutional layer
# Explaining each parameter:
#   input_dim, input_length - specifies that for the input, size will be 4 x 1000
#   nb_filter - defines number of filters for this convolutional layer, here 320
#   filter_length - Describes length of filter -> here filters are 4x26
#   border_mode - valid means that when we apply our 4x26 filters to the 4x1000 input image
#                 we will move it to the right until we get to position 1000 - 26, and we will
#                 not go any further(we don't go out of bounds, which we would do if border_mode="same")
#   activation - relu is the function we apply after getting the dot product of each filter weight with a
#                  region of the input data. Relu(x) = {x if x > 0, 0 if x <= 0}
#                   So in this specific application, where xi,j is the value of ith row, jth col in the input data and
#                   fa,b is one of the filter weights, we do relu(sum(xi,j*fa_b))
#   subsample_length - we say stride length is 1, meaning we move the filter over by 1 when applying it to input
model.add(Convolution1D(input_dim=4,
                        input_length=1000,
                        nb_filter=320,
                        filter_length=26,
                        border_mode="valid",
                        activation="relu",
                        subsample_length=1))

# We add a max pooling layer - it takes the 4x(1000-26) output of the convolutional layer
# and for each row it takes 13 values and returns the max, then slides window over by 13
# to take the max over 13 new values, then slides window over by 13, etc
# pool_length is number of outputs pooling window considers, stride is how much
# window moves over each iteration
model.add(MaxPooling1D(pool_length=13, stride=13))

# Dropout layer will randomly set, at a rate of 0.2(or 1 every 5), values of the data given to it
# to 0, and the rest of the values will be multiplied by 1/0.8(so the sum over values doesn't change)
# Dropout layers are useful for preventing overfitting i.e preventing the network from overlearning
# on the input data(overfitting occurs when its weights give high accuracy on inputs from the training data but on inputs
# that it has never seen, it doesn't do as well.
model.add(Dropout(0.2))

# add the bi directional lstm layer
# I believed the outputs of both forward and backward layer are stacked on top
# of each other and given as input into the next layer
model.add(brnn)

# Create an additional drop out layer, this time with rate of 0.5
model.add(Dropout(0.5))

# Take the matrix with rows stacked on each other, and output them
# placed one by one next to each other, all in the same row(hence "flattened")
model.add(Flatten())

# Add a fully connected layer which takes a 1x(75*640) input matrix and outputs
# a 1x925 matrix
model.add(Dense(input_dim=75*640, output_dim=925))
# Take the 1x925 matrix and on each individual cell apply
# RELU(x) = {x if x > 0, 0 if x <= 0}
model.add(Activation('relu'))

# Add a fully connected layer which takes a 1x925 input matrix and outputs
# a 1x919 matrix
model.add(Dense(input_dim=925, output_dim=919))
# Take the 919 matrix and on each individual cell apply
# Sigmoid(x) = 1/(1+e^-x)
model.add(Activation('sigmoid'))

print 'compiling model'
model.compile(loss='binary_crossentropy', optimizer='rmsprop', class_mode="binary")

print 'running at most 60 epochs'

checkpointer = ModelCheckpoint(filepath="DanQ_bestmodel.hdf5", verbose=1, save_best_only=True)
earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

model.fit(X_train, y_train, batch_size=100, nb_epoch=60, shuffle=True, show_accuracy=True, validation_data=(np.transpose(validmat['validxdata'],axes=(0,2,1)), validmat['validdata']), callbacks=[checkpointer,earlystopper])

tresults = model.evaluate(np.transpose(testmat['testxdata'],axes=(0,2,1)), testmat['testdata'],show_accuracy=True, return_dict=True)

print tresults
