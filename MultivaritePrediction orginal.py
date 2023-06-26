# Importing Libraries
# Generally measured in Mean Absolute Percentage Error (MAPE)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pandas import DataFrame
from pandas import concat
from numpy import array
from numpy import hstack
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv2D, MaxPooling2D, Flatten,Conv1D,Dropout
# saving and loading the .h5 model
from keras.models import Model
# define the model architecture
from keras.callbacks import Callback
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import Adam
from keras.layers import LSTM, Dropout, Dense, Input, Dot, Activation
from keras.layers import SimpleRNN
#####
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
# make the model reproducible Ask Luc
import time
import os
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Input
from keras.optimizers import Adam
from keras.losses import Huber
from keras import backend as K

np.random.seed(1)
tf.random.set_seed(2)
# FUNCTIONS
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

class Metric:
    def __init__(self):
        self.metrics = pd.DataFrame(columns=['R-squared', 'MAE', 'MSE',  'CVRMSE'])

    def add(self, y_true, y_pred):
        residuals = y_true - y_pred
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        mae = np.mean(np.abs(residuals))
        mse = np.mean(residuals ** 2)
        #mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        rmse = np.sqrt(mse)
        cvrmse = (rmse / np.mean(y_true)) * 100
        self.metrics.loc[len(self.metrics)] = [r_squared, mae, mse,  cvrmse]

    def get(self):
        return self.metrics

import matplotlib.pyplot as plt

class MetricsPlotter:
    def __init__(self, mse, val_mse, loss, val_loss):
        self.mse = mse
        self.val_mse = val_mse
        self.loss = loss
        self.val_loss = val_loss
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))

    def plot(self, title=None):
        epochs = len(self.mse)  # Get the number of epochs

        # plot the mse values on the first subplot
        self.ax1.plot(range(epochs), self.mse, label='Training MSE')
        self.ax1.plot(range(epochs), self.val_mse, label='Validation MSE')
        self.ax1.set_ylabel('MSE')
        self.ax1.legend()

        # plot the loss values on the second subplot
        self.ax2.plot(range(epochs), self.loss, label='Training Loss')
        self.ax2.plot(range(epochs), self.val_loss, label='Validation Loss')
        self.ax2.set_ylabel('Loss')
        self.ax2.legend()

        # set the x-axis label and tick marks
        self.ax2.set_xlabel('Epoch')
        self.ax2.set_xticks(range(epochs))
        self.ax2.set_xticklabels(range(1, epochs + 1))

        # set the title of the plot
        if title is not None:
            self.fig.suptitle(title)

        plt.show()

# make the results reproducible
def split_data2(data, first_fraction, second_fraction=None):
    n_total = len(data)
    n_first_split = round(n_total * first_fraction)

    first_split = data[:n_first_split]

    if second_fraction is None:
        second_split = data[n_first_split:]
    else:
        n_second_split = round(n_total * second_fraction)
        second_split = data[n_first_split:n_first_split+n_second_split]

    return first_split, second_split
def split_data(series):
    """Splits input series into train, val and test.
    """
    train_data = series[:26280]
    test_data = series[26280:35040]
    return train_data, test_data
# split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		# check if we are beyond the dataset
		if out_end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix:out_end_ix, -1]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

def R_squared(y_true, y_pred):
    residuals = y_true - y_pred
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2

# Data is splitted with thime (Miroslava)
def load_data(col=None, path="", verbose=False):
    df = pd.read_csv(path)
    if col is not None:
        df = df[col]
    if verbose:
        print(df.head())
    return df

def load_data(col=None,
              path='.\Kavgic Years.csv', verbose=False):
    df = pd.read_csv(path)
    if col is not None:
        df = df[col]
    if verbose:
        print(df.head())
    return df

from sklearn.preprocessing import MinMaxScaler

def normalize_data(dataframe):
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(dataframe)
    normalized_dataframe = pd.DataFrame(normalized_data, columns=dataframe.columns)
    return normalized_dataframe

# data with time as feature
imput_col_time = ['DryBulbTemperature','DirectNormalRadiation','OccupancySchedule',
                         'dayofw','Heating']
#imput_col_time = ['Dry Bulb Temperature','Direct Normal Radiation','Occupancy Schedule','dayofw','Heating']
multivar_df_time =load_data(imput_col_time,path='.\Kavgic Years.csv', verbose = True )
multivar10_19 = load_data(imput_col_time,path='.\dfall.csv',verbose = True )
multivar11_19 = load_data(imput_col_time,path='.\k2011-2019.csv',verbose = True )
multivar12_19 = load_data(imput_col_time,path='.\k2012-2019.csv',verbose = True )
multivar13_19 = load_data(imput_col_time,path='.\k2013-2019.csv',verbose = True )
multivar14_19 = load_data(imput_col_time,path='.\k2014-2019.csv',verbose = True )
multivar15_19 = load_data(imput_col_time,path='.\k2015-2019.csv',verbose = True )
multivar16_19 = load_data(imput_col_time,path='.\k2016-2019.csv',verbose = True )

List_S_fea = [multivar12_19,multivar13_19,multivar14_19, multivar15_19,
              multivar16_19 , multivar_df_time,multivar10_19,]

input_features = imput_col_time
for i in range(len(List_S_fea)):
    List_S_fea[i]['dayofw'] = List_S_fea[i]['dayofw'].astype(float)

# using trigonometry functions
R2_for_first_hour = []
times = []

s = List_S_fea[0]
train_multi, test_multi = split_data(s)
ep = 200
metric = Metric()

for j in s:
    def split_data(series):
        """Splits input series into train, val and test.
        """
        train_data = series[:61320] # 2011 start from 70080 done saved 11 , 61320 , 52560 , 43800
        test_data = series[61320:]
        return train_data, test_data
    vars()[f"in_seqt{j}"] = train_multi[j].values
    vars()[f"in_seqt{j}"].reshape((len(train_multi[j]), 1))
    vars()[f"in_seqt_test{j}"] = test_multi[j].values
    vars()[f"in_seqt_test{j}"].reshape((len(vars()[f"in_seqt_test{j}"]), 1))
out_seqt = in_seqtHeating
dataset = hstack((in_seqtDirectNormalRadiation.reshape(-1, 1),in_seqtDryBulbTemperature.reshape(-1, 1),
                  in_seqtOccupancySchedule.reshape(-1, 1), in_seqtdayofw.reshape(-1, 1),
                  in_seqtHeating.reshape(-1, 1), out_seqt.reshape(-1, 1)))
out_seqt_test = in_seqt_testHeating
dataset_test = hstack((in_seqt_testDirectNormalRadiation.reshape(-1, 1),
                       in_seqt_testDryBulbTemperature.reshape(-1, 1),
                       in_seqt_testOccupancySchedule.reshape(-1, 1), in_seqt_testdayofw.reshape(-1, 1),
                       in_seqt_testHeating.reshape(-1, 1), out_seqt_test.reshape(-1, 1)))
n_steps_in, n_steps_out = 16, 24
X, y = split_sequences(dataset, n_steps_in, n_steps_out)
X1, y1 = split_sequences(dataset_test, n_steps_in, n_steps_out)
n_features = X.shape[2]


# LSTM
start_time = time.time()
tf.keras.backend.clear_session()
model = Sequential()
model.add(LSTM(128, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
model.add(Dropout(0.3))
model.add(LSTM(128, activation='relu', return_sequences=True))
model.add(Dropout(0.3))
#model.add(LSTM(128, activation='relu', return_sequences=True))
#model.add(Dropout(0.3))
model.add(LSTM(128, activation='relu'))
model.add(Dense(n_steps_out, activation='relu'))
# set up the learning rate scheduler and early stopping
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=30, verbose=1, restore_best_weights=True)
# compile the model with the Adam optimizer and Huber loss
model.compile(optimizer=Adam(lr=1e-4), loss=tf.keras.losses.Huber(), metrics=['mse'])
# train the model with early stopping and the learning rate scheduler
history = model.fit(X, y, epochs=ep, verbose=1, validation_data=(X1, y1),
                    callbacks=[lr_scheduler, early_stopping])
# Get mae and loss from history log
mse = history.history['mse']
loss = history.history['loss']
# Get number of epochs
# get the mse and loss values from the history object
val_mse = history.history['val_mse']
val_loss = history.history['val_loss']
# save model
model.save('LSTM0.h5')

# PostProcessing
# create an instance of the MetricsPlotter class
metrics_plotter = MetricsPlotter(mse, val_mse, loss, val_loss)
metrics_plotter.plot(title='LSTM')
yhat1 = model.predict(X1, verbose=0)
num_cols = n_steps_out
# Generate column names
pred_cols = ['pt' + str(i + 1) for i in range(num_cols)]
real_cols = ['t' + str(i + 1) for i in range(num_cols)]
# Create dataframes
df_prediction = pd.DataFrame(yhat1, columns=pred_cols)
df_realvalues = pd.DataFrame(y1, columns=real_cols)
pred_1 = df_prediction['pt1']
Real_1 = df_realvalues['t1']
R2_for_first_hour.append(R_squared(Real_1, pred_1))
metric.add(y1, yhat1)
metrics_df = metric.get()
end_time = time.time()
times.append(end_time - start_time)

# CNN_lstm

# clear previous session
start_time = time.time()
tf.keras.backend.clear_session()
model_cnn_lstm = Sequential()
# add a convolutional layer with 64 filters, a kernel size of 2, and ReLU activation
model_cnn_lstm.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps_in, n_features)))
# add a max pooling layer with a pool size of 2
model_cnn_lstm.add(MaxPooling1D(pool_size=2))
# add a dropout layer with a rate of 0.3
model_cnn_lstm.add(Dropout(0.3))

# add an LSTM layer with 128 units and return sequences
model_cnn_lstm.add(LSTM(128, activation='relu', return_sequences=True))
# add another dropout layer with a rate of 0.3
model_cnn_lstm.add(Dropout(0.3))

# add a flatten layer to convert the output of the previous layer to a 1D tensor
model_cnn_lstm.add(Flatten())
# add a dense layer with 50 units and ReLU activation
model_cnn_lstm.add(Dense(50, activation='relu'))
# add a dense layer with n_steps_out units and ReLU activation
model_cnn_lstm.add(Dense(n_steps_out, activation='relu'))

# define the loss function as Huber
loss = tf.keras.losses.Huber()
# compile the model with adam optimizer and mse as a metric
model_cnn_lstm.compile(loss=loss, optimizer='adam', metrics=['mse'])

# define the learning rate scheduler and early stopping callbacks
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=30, verbose=1, restore_best_weights=True)

# print a summary of the model
model_cnn_lstm.summary()

# fit the model
history = model_cnn_lstm.fit(X, y, epochs=ep, verbose=1, validation_data=(X1, y1),
                             callbacks=[lr_scheduler, early_stopping])

# save the model
model_cnn_lstm.save('CNNLSTM.h5')

# get the mse and loss values from the history object
mse = history.history['mse']
loss = history.history['loss']
val_mse = history.history['val_mse']
val_loss = history.history['val_loss']
# PostProcessing
# create an instance of the MetricsPlotter class
metrics_plotter = MetricsPlotter(mse, val_mse, loss, val_loss)
metrics_plotter.plot(title='timeasfeature_CNNLSTM_past8andNext24DNN')
yhat1 = model_cnn_lstm.predict(X1, verbose=0)
num_cols = n_steps_out
# Generate column names
pred_cols = ['pt' + str(i + 1) for i in range(num_cols)]
real_cols = ['t' + str(i + 1) for i in range(num_cols)]
# Create dataframes
df_prediction = pd.DataFrame(yhat1, columns=pred_cols)
df_realvalues = pd.DataFrame(y1, columns=real_cols)
pred_1 = df_prediction['pt1']
Real_1 = df_realvalues['t1']
R2_for_first_hour.append(R_squared(Real_1, pred_1))
metric.add(y1, yhat1)
metrics_df = metric.get()
end_time = time.time()
times.append(end_time - start_time)

# RNN LSTM
start_time = time.time()
tf.keras.backend.clear_session()
# Define model architecture
hybrid_rnn_lstm_model = Sequential()
hybrid_rnn_lstm_model.add(
    Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps_in, n_features)))
hybrid_rnn_lstm_model.add(MaxPooling1D(pool_size=2))
hybrid_rnn_lstm_model.add(LSTM(64, activation='relu', return_sequences=True))
hybrid_rnn_lstm_model.add(Dropout(0.2))
hybrid_rnn_lstm_model.add(LSTM(128, activation='relu', return_sequences=True))
hybrid_rnn_lstm_model.add(Dropout(0.2))
hybrid_rnn_lstm_model.add(SimpleRNN(64, activation='relu', return_sequences=True))
hybrid_rnn_lstm_model.add(Dropout(0.2))
hybrid_rnn_lstm_model.add(SimpleRNN(32, activation='relu'))
hybrid_rnn_lstm_model.add(Dense(n_steps_out, activation='relu'))

# Compile the model
loss = tf.keras.losses.Huber()
hybrid_rnn_lstm_model.compile(loss=loss, optimizer='adam', metrics=['mse'])

# Define callbacks
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=30, verbose=1, restore_best_weights=True)

# Print model summary
hybrid_rnn_lstm_model.summary()

# Fit the model
history = hybrid_rnn_lstm_model.fit(X, y, epochs=ep, verbose=1, validation_data=(X1, y1),
                                    callbacks=[lr_scheduler, early_stopping])

# Get the mse and loss values from the history object
mse = history.history['mse']
loss = history.history['loss']
val_mse = history.history['val_mse']
val_loss = history.history['val_loss']

# Save the model
hybrid_rnn_lstm_model.save('RNNLSTM.h5')

# PostProcessing
# create an instance of the MetricsPlotter class
metrics_plotter = MetricsPlotter(mse, val_mse, loss, val_loss)
metrics_plotter.plot(title='RNN LSTM')
yhat1 = hybrid_rnn_lstm_model.predict(X1, verbose=0)
num_cols = n_steps_out
# Generate column names
pred_cols = ['pt' + str(i + 1) for i in range(num_cols)]
real_cols = ['t' + str(i + 1) for i in range(num_cols)]
# Create dataframes
df_prediction = pd.DataFrame(yhat1, columns=pred_cols)
df_realvalues = pd.DataFrame(y1, columns=real_cols)
pred_1 = df_prediction['pt1']
Real_1 = df_realvalues['t1']
R2_for_first_hour.append(R_squared(Real_1, pred_1))
metric.add(y1, yhat1)
metrics_df = metric.get()
end_time = time.time()
times.append(end_time - start_time)

# CNN RNN
# clear previous session
start_time = time.time()
tf.keras.backend.clear_session()
model_cnn_rnn = Sequential()

# add a convolutional layer with 64 filters, a kernel size of 2, and ReLU activation
model_cnn_rnn.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps_in, n_features)))
# add a max pooling layer with a pool size of 2
model_cnn_rnn.add(MaxPooling1D(pool_size=2))
# add a dropout layer with a rate of 0.3
model_cnn_rnn.add(Dropout(0.3))

# add a SimpleRNN layer with 128 units and return sequences
model_cnn_rnn.add(SimpleRNN(128, activation='relu', return_sequences=True))
# add another dropout layer with a rate of 0.3
model_cnn_rnn.add(Dropout(0.3))

# add a flatten layer to convert the output of the previous layer to a 1D tensor
model_cnn_rnn.add(Flatten())
# add a dense layer with 50 units and ReLU activation
model_cnn_rnn.add(Dense(50, activation='relu'))
# add a dense layer with n_steps_out units and ReLU activation
model_cnn_rnn.add(Dense(n_steps_out, activation='relu'))

# define the loss function as Huber
loss = tf.keras.losses.Huber()
# compile the model with adam optimizer and mse as a metric
model_cnn_rnn.compile(loss=loss, optimizer='adam', metrics=['mse'])

# define the learning rate scheduler and early stopping callbacks
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=30, verbose=1, restore_best_weights=True)

# print a summary of the model
model_cnn_rnn.summary()

# fit the model
history = model_cnn_rnn.fit(X, y, epochs=ep, verbose=1, validation_data=(X1, y1),
                            callbacks=[lr_scheduler, early_stopping])

# save the model
model_cnn_rnn.save('CNNRNN.h5')

# get the mse and loss values from the history object
mse = history.history['mse']
loss = history.history['loss']
val_mse = history.history['val_mse']
val_loss = history.history['val_loss']

# PostProcessing
# create an instance of the MetricsPlotter class
metrics_plotter = MetricsPlotter(mse, val_mse, loss, val_loss)
metrics_plotter.plot(title='CNNRNN')
yhat1 = model_cnn_rnn.predict(X1, verbose=0)
num_cols = n_steps_out
# Generate column names
pred_cols = ['pt' + str(i + 1) for i in range(num_cols)]
real_cols = ['t' + str(i + 1) for i in range(num_cols)]
# Create dataframes
df_prediction = pd.DataFrame(yhat1, columns=pred_cols)
df_realvalues = pd.DataFrame(y1, columns=real_cols)
pred_1 = df_prediction['pt1']
Real_1 = df_realvalues['t1']
R2_for_first_hour.append(R_squared(Real_1, pred_1))
metric.add(y1, yhat1)
metrics_df = metric.get()
end_time = time.time()
times.append(end_time - start_time)
# CNN
start_time = time.time()
tf.keras.backend.clear_session()
model_cnn = Sequential()
model_cnn.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps_in, n_features)))
model_cnn.add(MaxPooling1D(pool_size=2))
model_cnn.add(Flatten())
model_cnn.add(Dense(50, activation='relu'))
model_cnn.add(Dense(n_steps_out, activation='relu'))
loss = tf.keras.losses.Huber()
model_cnn.compile(loss=loss, optimizer='adam', metrics=['mse'])
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=30, verbose=1, restore_best_weights=True)
model_cnn.summary()
# fit model
history = model_cnn.fit(X, y, epochs=ep, verbose=1, validation_data=(X1, y1),
                        callbacks=[lr_scheduler, early_stopping])
# Get mae and loss from history log
mse = history.history['mse']
loss = history.history['loss']
# Get number of epochs
# get the mse and loss values from the history object
val_mse = history.history['val_mse']
val_loss = history.history['val_loss']
model_cnn.save('CNN.h5')

# PostProcessing
# create an instance of the MetricsPlotter class
metrics_plotter = MetricsPlotter(mse, val_mse, loss, val_loss)
metrics_plotter.plot(title='CNN')
yhat1 = model_cnn.predict(X1, verbose=0)
num_cols = n_steps_out
# Generate column names
pred_cols = ['pt' + str(i + 1) for i in range(num_cols)]
real_cols = ['t' + str(i + 1) for i in range(num_cols)]
# Create dataframes
df_prediction = pd.DataFrame(yhat1, columns=pred_cols)
df_realvalues = pd.DataFrame(y1, columns=real_cols)
pred_1 = df_prediction['pt1']
Real_1 = df_realvalues['t1']
R2_for_first_hour.append(R_squared(Real_1, pred_1))
metric.add(y1, yhat1)
metrics_df = metric.get()
end_time = time.time()
times.append(end_time - start_time)

# RNN
start_time = time.time()
tf.keras.backend.clear_session()
model_rnn = Sequential()
model_rnn.add(SimpleRNN(100, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
model_rnn.add(SimpleRNN(100, activation='relu'))
model_rnn.add(Dense(n_steps_out, activation='relu'))
loss = tf.keras.losses.Huber()
model_rnn.compile(loss=loss, optimizer='adam', metrics=['mse'])
model_rnn.summary()
# fit model
history = model_rnn.fit(X, y, epochs=ep, verbose=1, validation_data=(X1, y1),
                        callbacks=[lr_scheduler, early_stopping])
# Get mae and loss from history log
mse = history.history['mse']
loss = history.history['loss']
# Get number of epochs
# get the mse and loss values from the history object
val_mse = history.history['val_mse']
val_loss = history.history['val_loss']
model_rnn.save('RNN.h5')
# PostProcessing
# create an instance of the MetricsPlotter class
metrics_plotter = MetricsPlotter(mse, val_mse, loss, val_loss)
metrics_plotter.plot(title='RNN')
yhat1 = model_rnn.predict(X1, verbose=0)
num_cols = n_steps_out
# Generate column names
pred_cols = ['pt' + str(i + 1) for i in range(num_cols)]
real_cols = ['t' + str(i + 1) for i in range(num_cols)]
# Create dataframes
df_prediction = pd.DataFrame(yhat1, columns=pred_cols)
df_realvalues = pd.DataFrame(y1, columns=real_cols)
pred_1 = df_prediction['pt1']
Real_1 = df_realvalues['t1']
R2_for_first_hour.append(R_squared(Real_1, pred_1))
metric.add(y1, yhat1)
metrics_df = metric.get()
end_time = time.time()
times.append(end_time - start_time)
metrics_df.to_csv('metric', index=False)

import csv
# Specify the CSV file path
csv_file = 'Times.csv'

# Open the CSV file in write mode
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)

    # Iterate over the list and write each element as a row in the CSV file
    for item in times:
        writer.writerow([item])


s = List_S_fea[1]

for j in s:
    def split_data(series):
        """Splits input series into train, val and test.
        """
        train_data = series[:52560] # 2011 start from 78840 2010 , 70080 done saved 11 , 61320 , 52560 , 43800
        test_data = series[52560:]
        return train_data, test_data
    vars()[f"in_seqt{j}"] = train_multi[j].values
    vars()[f"in_seqt{j}"].reshape((len(train_multi[j]), 1))
    vars()[f"in_seqt_test{j}"] = test_multi[j].values
    vars()[f"in_seqt_test{j}"].reshape((len(vars()[f"in_seqt_test{j}"]), 1))
out_seqt = in_seqtHeating
dataset = hstack((in_seqtDirectNormalRadiation.reshape(-1, 1),in_seqtDryBulbTemperature.reshape(-1, 1),
                  in_seqtOccupancySchedule.reshape(-1, 1), in_seqtdayofw.reshape(-1, 1),
                  in_seqtHeating.reshape(-1, 1), out_seqt.reshape(-1, 1)))
out_seqt_test = in_seqt_testHeating
dataset_test = hstack((in_seqt_testDirectNormalRadiation.reshape(-1, 1),
                       in_seqt_testDryBulbTemperature.reshape(-1, 1),
                       in_seqt_testOccupancySchedule.reshape(-1, 1), in_seqt_testdayofw.reshape(-1, 1),
                       in_seqt_testHeating.reshape(-1, 1), out_seqt_test.reshape(-1, 1)))
n_steps_in, n_steps_out = 16, 24
X, y = split_sequences(dataset, n_steps_in, n_steps_out)
X1, y1 = split_sequences(dataset_test, n_steps_in, n_steps_out)
n_features = X.shape[2]

# LSTM
start_time = time.time()
tf.keras.backend.clear_session()
model = Sequential()
model.add(LSTM(128, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
model.add(Dropout(0.3))
model.add(LSTM(128, activation='relu', return_sequences=True))
model.add(Dropout(0.3))
#model.add(LSTM(128, activation='relu', return_sequences=True))
#model.add(Dropout(0.3))
model.add(LSTM(128, activation='relu'))
model.add(Dense(n_steps_out, activation='relu'))
# set up the learning rate scheduler and early stopping
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=30, verbose=1, restore_best_weights=True)
# compile the model with the Adam optimizer and Huber loss
model.compile(optimizer=Adam(lr=1e-4), loss=tf.keras.losses.Huber(), metrics=['mse'])
# train the model with early stopping and the learning rate scheduler
history = model.fit(X, y, epochs=ep, verbose=1, validation_data=(X1, y1),
                    callbacks=[lr_scheduler, early_stopping])
# Get mae and loss from history log
mse = history.history['mse']
loss = history.history['loss']
# Get number of epochs
# get the mse and loss values from the history object
val_mse = history.history['val_mse']
val_loss = history.history['val_loss']
# save model
model.save('LSTM1.h5')

# PostProcessing
# create an instance of the MetricsPlotter class
metrics_plotter = MetricsPlotter(mse, val_mse, loss, val_loss)
metrics_plotter.plot(title='LSTM')
yhat1 = model.predict(X1, verbose=0)
num_cols = n_steps_out
# Generate column names
pred_cols = ['pt' + str(i + 1) for i in range(num_cols)]
real_cols = ['t' + str(i + 1) for i in range(num_cols)]
# Create dataframes
df_prediction = pd.DataFrame(yhat1, columns=pred_cols)
df_realvalues = pd.DataFrame(y1, columns=real_cols)
pred_1 = df_prediction['pt1']
Real_1 = df_realvalues['t1']
R2_for_first_hour.append(R_squared(Real_1, pred_1))
metric.add(y1, yhat1)
metrics_df = metric.get()
end_time = time.time()
times.append(end_time - start_time)

# CNN_lstm

# clear previous session
start_time = time.time()
tf.keras.backend.clear_session()
model_cnn_lstm = Sequential()
# add a convolutional layer with 64 filters, a kernel size of 2, and ReLU activation
model_cnn_lstm.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps_in, n_features)))
# add a max pooling layer with a pool size of 2
model_cnn_lstm.add(MaxPooling1D(pool_size=2))
# add a dropout layer with a rate of 0.3
model_cnn_lstm.add(Dropout(0.3))

# add an LSTM layer with 128 units and return sequences
model_cnn_lstm.add(LSTM(128, activation='relu', return_sequences=True))
# add another dropout layer with a rate of 0.3
model_cnn_lstm.add(Dropout(0.3))

# add a flatten layer to convert the output of the previous layer to a 1D tensor
model_cnn_lstm.add(Flatten())
# add a dense layer with 50 units and ReLU activation
model_cnn_lstm.add(Dense(50, activation='relu'))
# add a dense layer with n_steps_out units and ReLU activation
model_cnn_lstm.add(Dense(n_steps_out, activation='relu'))

# define the loss function as Huber
loss = tf.keras.losses.Huber()
# compile the model with adam optimizer and mse as a metric
model_cnn_lstm.compile(loss=loss, optimizer='adam', metrics=['mse'])

# define the learning rate scheduler and early stopping callbacks
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=30, verbose=1, restore_best_weights=True)

# print a summary of the model
model_cnn_lstm.summary()

# fit the model
history = model_cnn_lstm.fit(X, y, epochs=ep, verbose=1, validation_data=(X1, y1),
                             callbacks=[lr_scheduler, early_stopping])

# save the model
model_cnn_lstm.save('CNNLSTM1.h5')

# get the mse and loss values from the history object
mse = history.history['mse']
loss = history.history['loss']
val_mse = history.history['val_mse']
val_loss = history.history['val_loss']
# PostProcessing
# create an instance of the MetricsPlotter class
metrics_plotter = MetricsPlotter(mse, val_mse, loss, val_loss)
metrics_plotter.plot(title='timeasfeature_CNNLSTM_past8andNext24DNN')
yhat1 = model_cnn_lstm.predict(X1, verbose=0)
num_cols = n_steps_out
# Generate column names
pred_cols = ['pt' + str(i + 1) for i in range(num_cols)]
real_cols = ['t' + str(i + 1) for i in range(num_cols)]
# Create dataframes
df_prediction = pd.DataFrame(yhat1, columns=pred_cols)
df_realvalues = pd.DataFrame(y1, columns=real_cols)
pred_1 = df_prediction['pt1']
Real_1 = df_realvalues['t1']
R2_for_first_hour.append(R_squared(Real_1, pred_1))
metric.add(y1, yhat1)
metrics_df = metric.get()
end_time = time.time()
times.append(end_time - start_time)

# RNN LSTM
start_time = time.time()
tf.keras.backend.clear_session()
# Define model architecture
hybrid_rnn_lstm_model = Sequential()
hybrid_rnn_lstm_model.add(
    Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps_in, n_features)))
hybrid_rnn_lstm_model.add(MaxPooling1D(pool_size=2))
hybrid_rnn_lstm_model.add(LSTM(64, activation='relu', return_sequences=True))
hybrid_rnn_lstm_model.add(Dropout(0.2))
hybrid_rnn_lstm_model.add(LSTM(128, activation='relu', return_sequences=True))
hybrid_rnn_lstm_model.add(Dropout(0.2))
hybrid_rnn_lstm_model.add(SimpleRNN(64, activation='relu', return_sequences=True))
hybrid_rnn_lstm_model.add(Dropout(0.2))
hybrid_rnn_lstm_model.add(SimpleRNN(32, activation='relu'))
hybrid_rnn_lstm_model.add(Dense(n_steps_out, activation='relu'))

# Compile the model
loss = tf.keras.losses.Huber()
hybrid_rnn_lstm_model.compile(loss=loss, optimizer='adam', metrics=['mse'])

# Define callbacks
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=30, verbose=1, restore_best_weights=True)

# Print model summary
hybrid_rnn_lstm_model.summary()

# Fit the model
history = hybrid_rnn_lstm_model.fit(X, y, epochs=ep, verbose=1, validation_data=(X1, y1),
                                    callbacks=[lr_scheduler, early_stopping])

# Get the mse and loss values from the history object
mse = history.history['mse']
loss = history.history['loss']
val_mse = history.history['val_mse']
val_loss = history.history['val_loss']

# Save the model
hybrid_rnn_lstm_model.save('RNNLSTM1.h5')

# PostProcessing
# create an instance of the MetricsPlotter class
metrics_plotter = MetricsPlotter(mse, val_mse, loss, val_loss)
metrics_plotter.plot(title='RNN LSTM')
yhat1 = hybrid_rnn_lstm_model.predict(X1, verbose=0)
num_cols = n_steps_out
# Generate column names
pred_cols = ['pt' + str(i + 1) for i in range(num_cols)]
real_cols = ['t' + str(i + 1) for i in range(num_cols)]
# Create dataframes
df_prediction = pd.DataFrame(yhat1, columns=pred_cols)
df_realvalues = pd.DataFrame(y1, columns=real_cols)
pred_1 = df_prediction['pt1']
Real_1 = df_realvalues['t1']
R2_for_first_hour.append(R_squared(Real_1, pred_1))
metric.add(y1, yhat1)
metrics_df = metric.get()
end_time = time.time()
times.append(end_time - start_time)

# CNN RNN
# clear previous session
start_time = time.time()
tf.keras.backend.clear_session()
model_cnn_rnn = Sequential()

# add a convolutional layer with 64 filters, a kernel size of 2, and ReLU activation
model_cnn_rnn.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps_in, n_features)))
# add a max pooling layer with a pool size of 2
model_cnn_rnn.add(MaxPooling1D(pool_size=2))
# add a dropout layer with a rate of 0.3
model_cnn_rnn.add(Dropout(0.3))

# add a SimpleRNN layer with 128 units and return sequences
model_cnn_rnn.add(SimpleRNN(128, activation='relu', return_sequences=True))
# add another dropout layer with a rate of 0.3
model_cnn_rnn.add(Dropout(0.3))

# add a flatten layer to convert the output of the previous layer to a 1D tensor
model_cnn_rnn.add(Flatten())
# add a dense layer with 50 units and ReLU activation
model_cnn_rnn.add(Dense(50, activation='relu'))
# add a dense layer with n_steps_out units and ReLU activation
model_cnn_rnn.add(Dense(n_steps_out, activation='relu'))

# define the loss function as Huber
loss = tf.keras.losses.Huber()
# compile the model with adam optimizer and mse as a metric
model_cnn_rnn.compile(loss=loss, optimizer='adam', metrics=['mse'])

# define the learning rate scheduler and early stopping callbacks
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=30, verbose=1, restore_best_weights=True)

# print a summary of the model
model_cnn_rnn.summary()

# fit the model
history = model_cnn_rnn.fit(X, y, epochs=ep, verbose=1, validation_data=(X1, y1),
                            callbacks=[lr_scheduler, early_stopping])

# save the model
model_cnn_rnn.save('CNNRNN1.h5')

# get the mse and loss values from the history object
mse = history.history['mse']
loss = history.history['loss']
val_mse = history.history['val_mse']
val_loss = history.history['val_loss']
# PostProcessing
# create an instance of the MetricsPlotter class
metrics_plotter = MetricsPlotter(mse, val_mse, loss, val_loss)
metrics_plotter.plot(title='CNNRNN')
yhat1 = model_cnn_rnn.predict(X1, verbose=0)
num_cols = n_steps_out
# Generate column names
pred_cols = ['pt' + str(i + 1) for i in range(num_cols)]
real_cols = ['t' + str(i + 1) for i in range(num_cols)]
# Create dataframes
df_prediction = pd.DataFrame(yhat1, columns=pred_cols)
df_realvalues = pd.DataFrame(y1, columns=real_cols)
pred_1 = df_prediction['pt1']
Real_1 = df_realvalues['t1']
R2_for_first_hour.append(R_squared(Real_1, pred_1))
metric.add(y1, yhat1)
metrics_df = metric.get()
end_time = time.time()
times.append(end_time - start_time)
# CNN
start_time = time.time()
tf.keras.backend.clear_session()
model_cnn = Sequential()
model_cnn.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps_in, n_features)))
model_cnn.add(MaxPooling1D(pool_size=2))
model_cnn.add(Flatten())
model_cnn.add(Dense(50, activation='relu'))
model_cnn.add(Dense(n_steps_out, activation='relu'))
loss = tf.keras.losses.Huber()
model_cnn.compile(loss=loss, optimizer='adam', metrics=['mse'])
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=30, verbose=1, restore_best_weights=True)
model_cnn.summary()
# fit model
history = model_cnn.fit(X, y, epochs=ep, verbose=1, validation_data=(X1, y1),
                        callbacks=[lr_scheduler, early_stopping])
# Get mae and loss from history log
mse = history.history['mse']
loss = history.history['loss']
# Get number of epochs
# get the mse and loss values from the history object
val_mse = history.history['val_mse']
val_loss = history.history['val_loss']
model_cnn.save('CNN1.h5')

# PostProcessing
# create an instance of the MetricsPlotter class
metrics_plotter = MetricsPlotter(mse, val_mse, loss, val_loss)
metrics_plotter.plot(title='CNN')
yhat1 = model_cnn.predict(X1, verbose=0)
num_cols = n_steps_out
# Generate column names
pred_cols = ['pt' + str(i + 1) for i in range(num_cols)]
real_cols = ['t' + str(i + 1) for i in range(num_cols)]
# Create dataframes
df_prediction = pd.DataFrame(yhat1, columns=pred_cols)
df_realvalues = pd.DataFrame(y1, columns=real_cols)
pred_1 = df_prediction['pt1']
Real_1 = df_realvalues['t1']
R2_for_first_hour.append(R_squared(Real_1, pred_1))
metric.add(y1, yhat1)
metrics_df = metric.get()
end_time = time.time()
times.append(end_time - start_time)

# RNN
start_time = time.time()
tf.keras.backend.clear_session()
model_rnn = Sequential()
model_rnn.add(SimpleRNN(100, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
model_rnn.add(SimpleRNN(100, activation='relu'))
model_rnn.add(Dense(n_steps_out, activation='relu'))
loss = tf.keras.losses.Huber()
model_rnn.compile(loss=loss, optimizer='adam', metrics=['mse'])
model_rnn.summary()
# fit model
history = model_rnn.fit(X, y, epochs=ep, verbose=1, validation_data=(X1, y1),
                        callbacks=[lr_scheduler, early_stopping])
# Get mae and loss from history log
mse = history.history['mse']
loss = history.history['loss']
# Get number of epochs
# get the mse and loss values from the history object
val_mse = history.history['val_mse']
val_loss = history.history['val_loss']
model_rnn.save('RNN1.h5')

# PostProcessing
# create an instance of the MetricsPlotter class
metrics_plotter = MetricsPlotter(mse, val_mse, loss, val_loss)
metrics_plotter.plot(title='RNN')
yhat1 = model_rnn.predict(X1, verbose=0)
num_cols = n_steps_out
# Generate column names
pred_cols = ['pt' + str(i + 1) for i in range(num_cols)]
real_cols = ['t' + str(i + 1) for i in range(num_cols)]
# Create dataframes
df_prediction = pd.DataFrame(yhat1, columns=pred_cols)
df_realvalues = pd.DataFrame(y1, columns=real_cols)
pred_1 = df_prediction['pt1']
Real_1 = df_realvalues['t1']
R2_for_first_hour.append(R_squared(Real_1, pred_1))
metric.add(y1, yhat1)
metrics_df = metric.get()
end_time = time.time()
times.append(end_time - start_time)
metrics_df.to_csv('metric1', index=False)

import csv
# Specify the CSV file path
csv_file = 'Times1.csv'

# Open the CSV file in write mode
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)

    # Iterate over the list and write each element as a row in the CSV file
    for item in times:
        writer.writerow([item])



s = List_S_fea[2]
for j in s:
    def split_data(series):
        """Splits input series into train, val and test.
        """
        train_data = series[:52560] # 2011 start from 70080 done saved 11 , 61320 , 52560 , 43800
        test_data = series[52560:]
        return train_data, test_data
    vars()[f"in_seqt{j}"] = train_multi[j].values
    vars()[f"in_seqt{j}"].reshape((len(train_multi[j]), 1))
    vars()[f"in_seqt_test{j}"] = test_multi[j].values
    vars()[f"in_seqt_test{j}"].reshape((len(vars()[f"in_seqt_test{j}"]), 1))
out_seqt = in_seqtHeating
dataset = hstack((in_seqtDirectNormalRadiation.reshape(-1, 1),in_seqtDryBulbTemperature.reshape(-1, 1),
                  in_seqtOccupancySchedule.reshape(-1, 1), in_seqtdayofw.reshape(-1, 1),
                  in_seqtHeating.reshape(-1, 1), out_seqt.reshape(-1, 1)))
out_seqt_test = in_seqt_testHeating
dataset_test = hstack((in_seqt_testDirectNormalRadiation.reshape(-1, 1),
                       in_seqt_testDryBulbTemperature.reshape(-1, 1),
                       in_seqt_testOccupancySchedule.reshape(-1, 1), in_seqt_testdayofw.reshape(-1, 1),
                       in_seqt_testHeating.reshape(-1, 1), out_seqt_test.reshape(-1, 1)))
n_steps_in, n_steps_out = 16, 24
X, y = split_sequences(dataset, n_steps_in, n_steps_out)
X1, y1 = split_sequences(dataset_test, n_steps_in, n_steps_out)
n_features = X.shape[2]


# LSTM
start_time = time.time()
tf.keras.backend.clear_session()
model = Sequential()
model.add(LSTM(128, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
model.add(Dropout(0.3))
model.add(LSTM(128, activation='relu', return_sequences=True))
model.add(Dropout(0.3))
#model.add(LSTM(128, activation='relu', return_sequences=True))
#model.add(Dropout(0.3))
model.add(LSTM(128, activation='relu'))
model.add(Dense(n_steps_out, activation='relu'))
# set up the learning rate scheduler and early stopping
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=30, verbose=1, restore_best_weights=True)
# compile the model with the Adam optimizer and Huber loss
model.compile(optimizer=Adam(lr=1e-4), loss=tf.keras.losses.Huber(), metrics=['mse'])
# train the model with early stopping and the learning rate scheduler
history = model.fit(X, y, epochs=ep, verbose=1, validation_data=(X1, y1),
                    callbacks=[lr_scheduler, early_stopping])
# Get mae and loss from history log
mse = history.history['mse']
loss = history.history['loss']
# Get number of epochs
# get the mse and loss values from the history object
val_mse = history.history['val_mse']
val_loss = history.history['val_loss']
# save model
model.save('LSTM2.h5')

# PostProcessing
# create an instance of the MetricsPlotter class
metrics_plotter = MetricsPlotter(mse, val_mse, loss, val_loss)
metrics_plotter.plot(title='LSTM')
yhat1 = model.predict(X1, verbose=0)
num_cols = n_steps_out
# Generate column names
pred_cols = ['pt' + str(i + 1) for i in range(num_cols)]
real_cols = ['t' + str(i + 1) for i in range(num_cols)]
# Create dataframes
df_prediction = pd.DataFrame(yhat1, columns=pred_cols)
df_realvalues = pd.DataFrame(y1, columns=real_cols)
pred_1 = df_prediction['pt1']
Real_1 = df_realvalues['t1']
R2_for_first_hour.append(R_squared(Real_1, pred_1))
metric.add(y1, yhat1)
metrics_df = metric.get()
end_time = time.time()
times.append(end_time - start_time)

# CNN_lstm

# clear previous session
start_time = time.time()
tf.keras.backend.clear_session()
model_cnn_lstm = Sequential()
# add a convolutional layer with 64 filters, a kernel size of 2, and ReLU activation
model_cnn_lstm.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps_in, n_features)))
# add a max pooling layer with a pool size of 2
model_cnn_lstm.add(MaxPooling1D(pool_size=2))
# add a dropout layer with a rate of 0.3
model_cnn_lstm.add(Dropout(0.3))

# add an LSTM layer with 128 units and return sequences
model_cnn_lstm.add(LSTM(128, activation='relu', return_sequences=True))
# add another dropout layer with a rate of 0.3
model_cnn_lstm.add(Dropout(0.3))

# add a flatten layer to convert the output of the previous layer to a 1D tensor
model_cnn_lstm.add(Flatten())
# add a dense layer with 50 units and ReLU activation
model_cnn_lstm.add(Dense(50, activation='relu'))
# add a dense layer with n_steps_out units and ReLU activation
model_cnn_lstm.add(Dense(n_steps_out, activation='relu'))

# define the loss function as Huber
loss = tf.keras.losses.Huber()
# compile the model with adam optimizer and mse as a metric
model_cnn_lstm.compile(loss=loss, optimizer='adam', metrics=['mse'])

# define the learning rate scheduler and early stopping callbacks
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=30, verbose=1, restore_best_weights=True)

# print a summary of the model
model_cnn_lstm.summary()

# fit the model
history = model_cnn_lstm.fit(X, y, epochs=ep, verbose=1, validation_data=(X1, y1),
                             callbacks=[lr_scheduler, early_stopping])

# save the model
model_cnn_lstm.save('CNNLSTM2.h5')

# get the mse and loss values from the history object
mse = history.history['mse']
loss = history.history['loss']
val_mse = history.history['val_mse']
val_loss = history.history['val_loss']
# PostProcessing
# create an instance of the MetricsPlotter class
metrics_plotter = MetricsPlotter(mse, val_mse, loss, val_loss)
metrics_plotter.plot(title='timeasfeature_CNNLSTM_past8andNext24DNN')
yhat1 = model_cnn_lstm.predict(X1, verbose=0)
num_cols = n_steps_out
# Generate column names
pred_cols = ['pt' + str(i + 1) for i in range(num_cols)]
real_cols = ['t' + str(i + 1) for i in range(num_cols)]
# Create dataframes
df_prediction = pd.DataFrame(yhat1, columns=pred_cols)
df_realvalues = pd.DataFrame(y1, columns=real_cols)
pred_1 = df_prediction['pt1']
Real_1 = df_realvalues['t1']
R2_for_first_hour.append(R_squared(Real_1, pred_1))
metric.add(y1, yhat1)
metrics_df = metric.get()
end_time = time.time()
times.append(end_time - start_time)

# RNN LSTM
start_time = time.time()
tf.keras.backend.clear_session()
# Define model architecture
hybrid_rnn_lstm_model = Sequential()
hybrid_rnn_lstm_model.add(
    Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps_in, n_features)))
hybrid_rnn_lstm_model.add(MaxPooling1D(pool_size=2))
hybrid_rnn_lstm_model.add(LSTM(64, activation='relu', return_sequences=True))
hybrid_rnn_lstm_model.add(Dropout(0.2))
hybrid_rnn_lstm_model.add(LSTM(128, activation='relu', return_sequences=True))
hybrid_rnn_lstm_model.add(Dropout(0.2))
hybrid_rnn_lstm_model.add(SimpleRNN(64, activation='relu', return_sequences=True))
hybrid_rnn_lstm_model.add(Dropout(0.2))
hybrid_rnn_lstm_model.add(SimpleRNN(32, activation='relu'))
hybrid_rnn_lstm_model.add(Dense(n_steps_out, activation='relu'))

# Compile the model
loss = tf.keras.losses.Huber()
hybrid_rnn_lstm_model.compile(loss=loss, optimizer='adam', metrics=['mse'])

# Define callbacks
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=30, verbose=1, restore_best_weights=True)

# Print model summary
hybrid_rnn_lstm_model.summary()

# Fit the model
history = hybrid_rnn_lstm_model.fit(X, y, epochs=ep, verbose=1, validation_data=(X1, y1),
                                    callbacks=[lr_scheduler, early_stopping])

# Get the mse and loss values from the history object
mse = history.history['mse']
loss = history.history['loss']
val_mse = history.history['val_mse']
val_loss = history.history['val_loss']

# Save the model
hybrid_rnn_lstm_model.save('RNNLSTM2.h5')

# PostProcessing
# create an instance of the MetricsPlotter class
metrics_plotter = MetricsPlotter(mse, val_mse, loss, val_loss)
metrics_plotter.plot(title='RNN LSTM')
yhat1 = hybrid_rnn_lstm_model.predict(X1, verbose=0)
num_cols = n_steps_out
# Generate column names
pred_cols = ['pt' + str(i + 1) for i in range(num_cols)]
real_cols = ['t' + str(i + 1) for i in range(num_cols)]
# Create dataframes
df_prediction = pd.DataFrame(yhat1, columns=pred_cols)
df_realvalues = pd.DataFrame(y1, columns=real_cols)
pred_1 = df_prediction['pt1']
Real_1 = df_realvalues['t1']
R2_for_first_hour.append(R_squared(Real_1, pred_1))
metric.add(y1, yhat1)
metrics_df = metric.get()
end_time = time.time()
times.append(end_time - start_time)

# CNN RNN
# clear previous session
start_time = time.time()
tf.keras.backend.clear_session()
model_cnn_rnn = Sequential()

# add a convolutional layer with 64 filters, a kernel size of 2, and ReLU activation
model_cnn_rnn.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps_in, n_features)))
# add a max pooling layer with a pool size of 2
model_cnn_rnn.add(MaxPooling1D(pool_size=2))
# add a dropout layer with a rate of 0.3
model_cnn_rnn.add(Dropout(0.3))

# add a SimpleRNN layer with 128 units and return sequences
model_cnn_rnn.add(SimpleRNN(128, activation='relu', return_sequences=True))
# add another dropout layer with a rate of 0.3
model_cnn_rnn.add(Dropout(0.3))

# add a flatten layer to convert the output of the previous layer to a 1D tensor
model_cnn_rnn.add(Flatten())
# add a dense layer with 50 units and ReLU activation
model_cnn_rnn.add(Dense(50, activation='relu'))
# add a dense layer with n_steps_out units and ReLU activation
model_cnn_rnn.add(Dense(n_steps_out, activation='relu'))

# define the loss function as Huber
loss = tf.keras.losses.Huber()
# compile the model with adam optimizer and mse as a metric
model_cnn_rnn.compile(loss=loss, optimizer='adam', metrics=['mse'])

# define the learning rate scheduler and early stopping callbacks
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=30, verbose=1, restore_best_weights=True)

# print a summary of the model
model_cnn_rnn.summary()

# fit the model
history = model_cnn_rnn.fit(X, y, epochs=ep, verbose=1, validation_data=(X1, y1),
                            callbacks=[lr_scheduler, early_stopping])

# save the model
model_cnn_rnn.save('CNNRNN2.h5')

# get the mse and loss values from the history object
mse = history.history['mse']
loss = history.history['loss']
val_mse = history.history['val_mse']
val_loss = history.history['val_loss']
# PostProcessing
# create an instance of the MetricsPlotter class
metrics_plotter = MetricsPlotter(mse, val_mse, loss, val_loss)
metrics_plotter.plot(title='CNNRNN')
yhat1 = model_cnn_rnn.predict(X1, verbose=0)
num_cols = n_steps_out
# Generate column names
pred_cols = ['pt' + str(i + 1) for i in range(num_cols)]
real_cols = ['t' + str(i + 1) for i in range(num_cols)]
# Create dataframes
df_prediction = pd.DataFrame(yhat1, columns=pred_cols)
df_realvalues = pd.DataFrame(y1, columns=real_cols)
pred_1 = df_prediction['pt1']
Real_1 = df_realvalues['t1']
R2_for_first_hour.append(R_squared(Real_1, pred_1))
metric.add(y1, yhat1)
metrics_df = metric.get()
end_time = time.time()
times.append(end_time - start_time)
# CNN
start_time = time.time()
tf.keras.backend.clear_session()
model_cnn = Sequential()
model_cnn.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps_in, n_features)))
model_cnn.add(MaxPooling1D(pool_size=2))
model_cnn.add(Flatten())
model_cnn.add(Dense(50, activation='relu'))
model_cnn.add(Dense(n_steps_out, activation='relu'))
loss = tf.keras.losses.Huber()
model_cnn.compile(loss=loss, optimizer='adam', metrics=['mse'])
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=30, verbose=1, restore_best_weights=True)
model_cnn.summary()
# fit model
history = model_cnn.fit(X, y, epochs=ep, verbose=1, validation_data=(X1, y1),
                        callbacks=[lr_scheduler, early_stopping])
# Get mae and loss from history log
mse = history.history['mse']
loss = history.history['loss']
# Get number of epochs
# get the mse and loss values from the history object
val_mse = history.history['val_mse']
val_loss = history.history['val_loss']
model_cnn.save('CNN2.h5')

# PostProcessing
# create an instance of the MetricsPlotter class
metrics_plotter = MetricsPlotter(mse, val_mse, loss, val_loss)
metrics_plotter.plot(title='CNN')
yhat1 = model_cnn.predict(X1, verbose=0)
num_cols = n_steps_out
# Generate column names
pred_cols = ['pt' + str(i + 1) for i in range(num_cols)]
real_cols = ['t' + str(i + 1) for i in range(num_cols)]
# Create dataframes
df_prediction = pd.DataFrame(yhat1, columns=pred_cols)
df_realvalues = pd.DataFrame(y1, columns=real_cols)
pred_1 = df_prediction['pt1']
Real_1 = df_realvalues['t1']
R2_for_first_hour.append(R_squared(Real_1, pred_1))
metric.add(y1, yhat1)
metrics_df = metric.get()
end_time = time.time()
times.append(end_time - start_time)

# RNN
start_time = time.time()
tf.keras.backend.clear_session()
model_rnn = Sequential()
model_rnn.add(SimpleRNN(100, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
model_rnn.add(SimpleRNN(100, activation='relu'))
model_rnn.add(Dense(n_steps_out, activation='relu'))
loss = tf.keras.losses.Huber()
model_rnn.compile(loss=loss, optimizer='adam', metrics=['mse'])
model_rnn.summary()
# fit model
history = model_rnn.fit(X, y, epochs=ep, verbose=1, validation_data=(X1, y1),
                        callbacks=[lr_scheduler, early_stopping])
# Get mae and loss from history log
mse = history.history['mse']
loss = history.history['loss']
# Get number of epochs
# get the mse and loss values from the history object
val_mse = history.history['val_mse']
val_loss = history.history['val_loss']
model_rnn.save('RNN2.h5')

# PostProcessing
# create an instance of the MetricsPlotter class
metrics_plotter = MetricsPlotter(mse, val_mse, loss, val_loss)
metrics_plotter.plot(title='RNN')
yhat1 = model_rnn.predict(X1, verbose=0)
num_cols = n_steps_out
# Generate column names
pred_cols = ['pt' + str(i + 1) for i in range(num_cols)]
real_cols = ['t' + str(i + 1) for i in range(num_cols)]
# Create dataframes
df_prediction = pd.DataFrame(yhat1, columns=pred_cols)
df_realvalues = pd.DataFrame(y1, columns=real_cols)
pred_1 = df_prediction['pt1']
Real_1 = df_realvalues['t1']
R2_for_first_hour.append(R_squared(Real_1, pred_1))
metric.add(y1, yhat1)
metrics_df = metric.get()
end_time = time.time()
times.append(end_time - start_time)
metrics_df.to_csv('metric2', index=False)

import csv
# Specify the CSV file path
csv_file = 'Times2.csv'

# Open the CSV file in write mode
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)

    # Iterate over the list and write each element as a row in the CSV file
    for item in times:
        writer.writerow([item])




s = List_S_fea[3]
for j in s:
    def split_data(series):
        """Splits input series into train, val and test.
        """
        train_data = series[:43800]  # 2011 start from 70080 done saved 11 , 61320 , 52560 , 43800
        test_data = series[43800:]
        return train_data, test_data


    vars()[f"in_seqt{j}"] = train_multi[j].values
    vars()[f"in_seqt{j}"].reshape((len(train_multi[j]), 1))
    vars()[f"in_seqt_test{j}"] = test_multi[j].values
    vars()[f"in_seqt_test{j}"].reshape((len(vars()[f"in_seqt_test{j}"]), 1))
out_seqt = in_seqtHeating
dataset = hstack((in_seqtDirectNormalRadiation.reshape(-1, 1), in_seqtDryBulbTemperature.reshape(-1, 1),
                  in_seqtOccupancySchedule.reshape(-1, 1), in_seqtdayofw.reshape(-1, 1),
                  in_seqtHeating.reshape(-1, 1), out_seqt.reshape(-1, 1)))
out_seqt_test = in_seqt_testHeating
dataset_test = hstack((in_seqt_testDirectNormalRadiation.reshape(-1, 1),
                       in_seqt_testDryBulbTemperature.reshape(-1, 1),
                       in_seqt_testOccupancySchedule.reshape(-1, 1), in_seqt_testdayofw.reshape(-1, 1),
                       in_seqt_testHeating.reshape(-1, 1), out_seqt_test.reshape(-1, 1)))
n_steps_in, n_steps_out = 16, 24
X, y = split_sequences(dataset, n_steps_in, n_steps_out)
X1, y1 = split_sequences(dataset_test, n_steps_in, n_steps_out)
n_features = X.shape[2]

# LSTM
start_time = time.time()
tf.keras.backend.clear_session()
model = Sequential()
model.add(LSTM(128, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
model.add(Dropout(0.3))
model.add(LSTM(128, activation='relu', return_sequences=True))
model.add(Dropout(0.3))
# model.add(LSTM(128, activation='relu', return_sequences=True))
# model.add(Dropout(0.3))
model.add(LSTM(128, activation='relu'))
model.add(Dense(n_steps_out, activation='relu'))
# set up the learning rate scheduler and early stopping
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=30, verbose=1, restore_best_weights=True)
# compile the model with the Adam optimizer and Huber loss
model.compile(optimizer=Adam(lr=1e-4), loss=tf.keras.losses.Huber(), metrics=['mse'])
# train the model with early stopping and the learning rate scheduler
history = model.fit(X, y, epochs=ep, verbose=1, validation_data=(X1, y1),
                    callbacks=[lr_scheduler, early_stopping])
# Get mae and loss from history log
mse = history.history['mse']
loss = history.history['loss']
# Get number of epochs
# get the mse and loss values from the history object
val_mse = history.history['val_mse']
val_loss = history.history['val_loss']
# save model
model.save('LSTM3.h5')

# PostProcessing
# create an instance of the MetricsPlotter class
metrics_plotter = MetricsPlotter(mse, val_mse, loss, val_loss)
metrics_plotter.plot(title='LSTM')
yhat1 = model.predict(X1, verbose=0)
num_cols = n_steps_out
# Generate column names
pred_cols = ['pt' + str(i + 1) for i in range(num_cols)]
real_cols = ['t' + str(i + 1) for i in range(num_cols)]
# Create dataframes
df_prediction = pd.DataFrame(yhat1, columns=pred_cols)
df_realvalues = pd.DataFrame(y1, columns=real_cols)
pred_1 = df_prediction['pt1']
Real_1 = df_realvalues['t1']
R2_for_first_hour.append(R_squared(Real_1, pred_1))
metric.add(y1, yhat1)
metrics_df = metric.get()
end_time = time.time()
times.append(end_time - start_time)

# CNN_lstm

# clear previous session
start_time = time.time()
tf.keras.backend.clear_session()
model_cnn_lstm = Sequential()
# add a convolutional layer with 64 filters, a kernel size of 2, and ReLU activation
model_cnn_lstm.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps_in, n_features)))
# add a max pooling layer with a pool size of 2
model_cnn_lstm.add(MaxPooling1D(pool_size=2))
# add a dropout layer with a rate of 0.3
model_cnn_lstm.add(Dropout(0.3))

# add an LSTM layer with 128 units and return sequences
model_cnn_lstm.add(LSTM(128, activation='relu', return_sequences=True))
# add another dropout layer with a rate of 0.3
model_cnn_lstm.add(Dropout(0.3))

# add a flatten layer to convert the output of the previous layer to a 1D tensor
model_cnn_lstm.add(Flatten())
# add a dense layer with 50 units and ReLU activation
model_cnn_lstm.add(Dense(50, activation='relu'))
# add a dense layer with n_steps_out units and ReLU activation
model_cnn_lstm.add(Dense(n_steps_out, activation='relu'))

# define the loss function as Huber
loss = tf.keras.losses.Huber()
# compile the model with adam optimizer and mse as a metric
model_cnn_lstm.compile(loss=loss, optimizer='adam', metrics=['mse'])

# define the learning rate scheduler and early stopping callbacks
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=30, verbose=1, restore_best_weights=True)

# print a summary of the model
model_cnn_lstm.summary()

# fit the model
history = model_cnn_lstm.fit(X, y, epochs=ep, verbose=1, validation_data=(X1, y1),
                             callbacks=[lr_scheduler, early_stopping])

# save the model
model_cnn_lstm.save('CNNLSTM3.h5')

# get the mse and loss values from the history object
mse = history.history['mse']
loss = history.history['loss']
val_mse = history.history['val_mse']
val_loss = history.history['val_loss']
# PostProcessing
# create an instance of the MetricsPlotter class
metrics_plotter = MetricsPlotter(mse, val_mse, loss, val_loss)
metrics_plotter.plot(title='timeasfeature_CNNLSTM_past8andNext24DNN')
yhat1 = model_cnn_lstm.predict(X1, verbose=0)
num_cols = n_steps_out
# Generate column names
pred_cols = ['pt' + str(i + 1) for i in range(num_cols)]
real_cols = ['t' + str(i + 1) for i in range(num_cols)]
# Create dataframes
df_prediction = pd.DataFrame(yhat1, columns=pred_cols)
df_realvalues = pd.DataFrame(y1, columns=real_cols)
pred_1 = df_prediction['pt1']
Real_1 = df_realvalues['t1']
R2_for_first_hour.append(R_squared(Real_1, pred_1))
metric.add(y1, yhat1)
metrics_df = metric.get()
end_time = time.time()
times.append(end_time - start_time)

# RNN LSTM
start_time = time.time()
tf.keras.backend.clear_session()
# Define model architecture
hybrid_rnn_lstm_model = Sequential()
hybrid_rnn_lstm_model.add(
    Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps_in, n_features)))
hybrid_rnn_lstm_model.add(MaxPooling1D(pool_size=2))
hybrid_rnn_lstm_model.add(LSTM(64, activation='relu', return_sequences=True))
hybrid_rnn_lstm_model.add(Dropout(0.2))
hybrid_rnn_lstm_model.add(LSTM(128, activation='relu', return_sequences=True))
hybrid_rnn_lstm_model.add(Dropout(0.2))
hybrid_rnn_lstm_model.add(SimpleRNN(64, activation='relu', return_sequences=True))
hybrid_rnn_lstm_model.add(Dropout(0.2))
hybrid_rnn_lstm_model.add(SimpleRNN(32, activation='relu'))
hybrid_rnn_lstm_model.add(Dense(n_steps_out, activation='relu'))

# Compile the model
loss = tf.keras.losses.Huber()
hybrid_rnn_lstm_model.compile(loss=loss, optimizer='adam', metrics=['mse'])

# Define callbacks
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=30, verbose=1, restore_best_weights=True)

# Print model summary
hybrid_rnn_lstm_model.summary()

# Fit the model
history = hybrid_rnn_lstm_model.fit(X, y, epochs=ep, verbose=1, validation_data=(X1, y1),
                                    callbacks=[lr_scheduler, early_stopping])

# Get the mse and loss values from the history object
mse = history.history['mse']
loss = history.history['loss']
val_mse = history.history['val_mse']
val_loss = history.history['val_loss']

# Save the model
hybrid_rnn_lstm_model.save('RNNLSTM3.h5')

# PostProcessing
# create an instance of the MetricsPlotter class
metrics_plotter = MetricsPlotter(mse, val_mse, loss, val_loss)
metrics_plotter.plot(title='RNN LSTM')
yhat1 = hybrid_rnn_lstm_model.predict(X1, verbose=0)
num_cols = n_steps_out
# Generate column names
pred_cols = ['pt' + str(i + 1) for i in range(num_cols)]
real_cols = ['t' + str(i + 1) for i in range(num_cols)]
# Create dataframes
df_prediction = pd.DataFrame(yhat1, columns=pred_cols)
df_realvalues = pd.DataFrame(y1, columns=real_cols)
pred_1 = df_prediction['pt1']
Real_1 = df_realvalues['t1']
R2_for_first_hour.append(R_squared(Real_1, pred_1))
metric.add(y1, yhat1)
metrics_df = metric.get()
end_time = time.time()
times.append(end_time - start_time)

# CNN RNN
# clear previous session
start_time = time.time()
tf.keras.backend.clear_session()
model_cnn_rnn = Sequential()

# add a convolutional layer with 64 filters, a kernel size of 2, and ReLU activation
model_cnn_rnn.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps_in, n_features)))
# add a max pooling layer with a pool size of 2
model_cnn_rnn.add(MaxPooling1D(pool_size=2))
# add a dropout layer with a rate of 0.3
model_cnn_rnn.add(Dropout(0.3))

# add a SimpleRNN layer with 128 units and return sequences
model_cnn_rnn.add(SimpleRNN(128, activation='relu', return_sequences=True))
# add another dropout layer with a rate of 0.3
model_cnn_rnn.add(Dropout(0.3))

# add a flatten layer to convert the output of the previous layer to a 1D tensor
model_cnn_rnn.add(Flatten())
# add a dense layer with 50 units and ReLU activation
model_cnn_rnn.add(Dense(50, activation='relu'))
# add a dense layer with n_steps_out units and ReLU activation
model_cnn_rnn.add(Dense(n_steps_out, activation='relu'))

# define the loss function as Huber
loss = tf.keras.losses.Huber()
# compile the model with adam optimizer and mse as a metric
model_cnn_rnn.compile(loss=loss, optimizer='adam', metrics=['mse'])

# define the learning rate scheduler and early stopping callbacks
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=30, verbose=1, restore_best_weights=True)

# print a summary of the model
model_cnn_rnn.summary()

# fit the model
history = model_cnn_rnn.fit(X, y, epochs=ep, verbose=1, validation_data=(X1, y1),
                            callbacks=[lr_scheduler, early_stopping])

# save the model
model_cnn_rnn.save('CNNRNN3.h5')

# get the mse and loss values from the history object
mse = history.history['mse']
loss = history.history['loss']
val_mse = history.history['val_mse']
val_loss = history.history['val_loss']
# PostProcessing
# create an instance of the MetricsPlotter class
metrics_plotter = MetricsPlotter(mse, val_mse, loss, val_loss)
metrics_plotter.plot(title='CNNRNN')
yhat1 = model_cnn_rnn.predict(X1, verbose=0)
num_cols = n_steps_out
# Generate column names
pred_cols = ['pt' + str(i + 1) for i in range(num_cols)]
real_cols = ['t' + str(i + 1) for i in range(num_cols)]
# Create dataframes
df_prediction = pd.DataFrame(yhat1, columns=pred_cols)
df_realvalues = pd.DataFrame(y1, columns=real_cols)
pred_1 = df_prediction['pt1']
Real_1 = df_realvalues['t1']
R2_for_first_hour.append(R_squared(Real_1, pred_1))
metric.add(y1, yhat1)
metrics_df = metric.get()
end_time = time.time()
times.append(end_time - start_time)
# CNN
start_time = time.time()
tf.keras.backend.clear_session()
model_cnn = Sequential()
model_cnn.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps_in, n_features)))
model_cnn.add(MaxPooling1D(pool_size=2))
model_cnn.add(Flatten())
model_cnn.add(Dense(50, activation='relu'))
model_cnn.add(Dense(n_steps_out, activation='relu'))
loss = tf.keras.losses.Huber()
model_cnn.compile(loss=loss, optimizer='adam', metrics=['mse'])
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=30, verbose=1, restore_best_weights=True)
model_cnn.summary()
# fit model
history = model_cnn.fit(X, y, epochs=ep, verbose=1, validation_data=(X1, y1),
                        callbacks=[lr_scheduler, early_stopping])
# Get mae and loss from history log
mse = history.history['mse']
loss = history.history['loss']
# Get number of epochs
# get the mse and loss values from the history object
val_mse = history.history['val_mse']
val_loss = history.history['val_loss']
model_cnn.save('CNN3.h5')

# PostProcessing
# create an instance of the MetricsPlotter class
metrics_plotter = MetricsPlotter(mse, val_mse, loss, val_loss)
metrics_plotter.plot(title='CNN')
yhat1 = model_cnn.predict(X1, verbose=0)
num_cols = n_steps_out
# Generate column names
pred_cols = ['pt' + str(i + 1) for i in range(num_cols)]
real_cols = ['t' + str(i + 1) for i in range(num_cols)]
# Create dataframes
df_prediction = pd.DataFrame(yhat1, columns=pred_cols)
df_realvalues = pd.DataFrame(y1, columns=real_cols)
pred_1 = df_prediction['pt1']
Real_1 = df_realvalues['t1']
R2_for_first_hour.append(R_squared(Real_1, pred_1))
metric.add(y1, yhat1)
metrics_df = metric.get()
end_time = time.time()
times.append(end_time - start_time)

# RNN
start_time = time.time()
tf.keras.backend.clear_session()
model_rnn = Sequential()
model_rnn.add(SimpleRNN(100, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
model_rnn.add(SimpleRNN(100, activation='relu'))
model_rnn.add(Dense(n_steps_out, activation='relu'))
loss = tf.keras.losses.Huber()
model_rnn.compile(loss=loss, optimizer='adam', metrics=['mse'])
model_rnn.summary()
# fit model
history = model_rnn.fit(X, y, epochs=ep, verbose=1, validation_data=(X1, y1),
                        callbacks=[lr_scheduler, early_stopping])
# Get mae and loss from history log
mse = history.history['mse']
loss = history.history['loss']
# Get number of epochs
# get the mse and loss values from the history object
val_mse = history.history['val_mse']
val_loss = history.history['val_loss']
model_rnn.save('RNN3.h5')

# PostProcessing
# create an instance of the MetricsPlotter class
metrics_plotter = MetricsPlotter(mse, val_mse, loss, val_loss)
metrics_plotter.plot(title='RNN')
yhat1 = model_rnn.predict(X1, verbose=0)
num_cols = n_steps_out
# Generate column names
pred_cols = ['pt' + str(i + 1) for i in range(num_cols)]
real_cols = ['t' + str(i + 1) for i in range(num_cols)]
# Create dataframes
df_prediction = pd.DataFrame(yhat1, columns=pred_cols)
df_realvalues = pd.DataFrame(y1, columns=real_cols)
pred_1 = df_prediction['pt1']
Real_1 = df_realvalues['t1']
R2_for_first_hour.append(R_squared(Real_1, pred_1))
metric.add(y1, yhat1)
metrics_df = metric.get()
end_time = time.time()
times.append(end_time - start_time)
metrics_df.to_csv('metric3', index=False)

import csv

# Specify the CSV file path
csv_file = 'Times3.csv'

# Open the CSV file in write mode
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)

    # Iterate over the list and write each element as a row in the CSV file
    for item in times:
        writer.writerow([item])



s = List_S_fea[4]

for j in s:
    def split_data(series):
        """Splits input series into train, val and test.
        """
        train_data = series[:35040]  # 2011 start from 70080 done saved 11 , 61320 , 52560 , 43800
        test_data = series[35040:]
        return train_data, test_data


    vars()[f"in_seqt{j}"] = train_multi[j].values
    vars()[f"in_seqt{j}"].reshape((len(train_multi[j]), 1))
    vars()[f"in_seqt_test{j}"] = test_multi[j].values
    vars()[f"in_seqt_test{j}"].reshape((len(vars()[f"in_seqt_test{j}"]), 1))
out_seqt = in_seqtHeating
dataset = hstack((in_seqtDirectNormalRadiation.reshape(-1, 1), in_seqtDryBulbTemperature.reshape(-1, 1),
                  in_seqtOccupancySchedule.reshape(-1, 1), in_seqtdayofw.reshape(-1, 1),
                  in_seqtHeating.reshape(-1, 1), out_seqt.reshape(-1, 1)))
out_seqt_test = in_seqt_testHeating
dataset_test = hstack((in_seqt_testDirectNormalRadiation.reshape(-1, 1),
                       in_seqt_testDryBulbTemperature.reshape(-1, 1),
                       in_seqt_testOccupancySchedule.reshape(-1, 1), in_seqt_testdayofw.reshape(-1, 1),
                       in_seqt_testHeating.reshape(-1, 1), out_seqt_test.reshape(-1, 1)))
n_steps_in, n_steps_out = 16, 24
X, y = split_sequences(dataset, n_steps_in, n_steps_out)
X1, y1 = split_sequences(dataset_test, n_steps_in, n_steps_out)
n_features = X.shape[2]

# LSTM
start_time = time.time()
tf.keras.backend.clear_session()
model = Sequential()
model.add(LSTM(128, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
model.add(Dropout(0.3))
model.add(LSTM(128, activation='relu', return_sequences=True))
model.add(Dropout(0.3))
# model.add(LSTM(128, activation='relu', return_sequences=True))
# model.add(Dropout(0.3))
model.add(LSTM(128, activation='relu'))
model.add(Dense(n_steps_out, activation='relu'))
# set up the learning rate scheduler and early stopping
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=30, verbose=1, restore_best_weights=True)
# compile the model with the Adam optimizer and Huber loss
model.compile(optimizer=Adam(lr=1e-4), loss=tf.keras.losses.Huber(), metrics=['mse'])
# train the model with early stopping and the learning rate scheduler
history = model.fit(X, y, epochs=ep, verbose=1, validation_data=(X1, y1),
                    callbacks=[lr_scheduler, early_stopping])
# Get mae and loss from history log
mse = history.history['mse']
loss = history.history['loss']
# Get number of epochs
# get the mse and loss values from the history object
val_mse = history.history['val_mse']
val_loss = history.history['val_loss']
# save model
model.save('LSTM4.h5')

# PostProcessing
# create an instance of the MetricsPlotter class
metrics_plotter = MetricsPlotter(mse, val_mse, loss, val_loss)
metrics_plotter.plot(title='LSTM')
yhat1 = model.predict(X1, verbose=0)
num_cols = n_steps_out
# Generate column names
pred_cols = ['pt' + str(i + 1) for i in range(num_cols)]
real_cols = ['t' + str(i + 1) for i in range(num_cols)]
# Create dataframes
df_prediction = pd.DataFrame(yhat1, columns=pred_cols)
df_realvalues = pd.DataFrame(y1, columns=real_cols)
pred_1 = df_prediction['pt1']
Real_1 = df_realvalues['t1']
R2_for_first_hour.append(R_squared(Real_1, pred_1))
metric.add(y1, yhat1)
metrics_df = metric.get()
end_time = time.time()
times.append(end_time - start_time)

# CNN_lstm

# clear previous session
start_time = time.time()
tf.keras.backend.clear_session()
model_cnn_lstm = Sequential()
# add a convolutional layer with 64 filters, a kernel size of 2, and ReLU activation
model_cnn_lstm.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps_in, n_features)))
# add a max pooling layer with a pool size of 2
model_cnn_lstm.add(MaxPooling1D(pool_size=2))
# add a dropout layer with a rate of 0.3
model_cnn_lstm.add(Dropout(0.3))

# add an LSTM layer with 128 units and return sequences
model_cnn_lstm.add(LSTM(128, activation='relu', return_sequences=True))
# add another dropout layer with a rate of 0.3
model_cnn_lstm.add(Dropout(0.3))

# add a flatten layer to convert the output of the previous layer to a 1D tensor
model_cnn_lstm.add(Flatten())
# add a dense layer with 50 units and ReLU activation
model_cnn_lstm.add(Dense(50, activation='relu'))
# add a dense layer with n_steps_out units and ReLU activation
model_cnn_lstm.add(Dense(n_steps_out, activation='relu'))

# define the loss function as Huber
loss = tf.keras.losses.Huber()
# compile the model with adam optimizer and mse as a metric
model_cnn_lstm.compile(loss=loss, optimizer='adam', metrics=['mse'])

# define the learning rate scheduler and early stopping callbacks
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=30, verbose=1, restore_best_weights=True)

# print a summary of the model
model_cnn_lstm.summary()

# fit the model
history = model_cnn_lstm.fit(X, y, epochs=ep, verbose=1, validation_data=(X1, y1),
                             callbacks=[lr_scheduler, early_stopping])

# save the model
model_cnn_lstm.save('CNNLSTM4.h5')

# get the mse and loss values from the history object
mse = history.history['mse']
loss = history.history['loss']
val_mse = history.history['val_mse']
val_loss = history.history['val_loss']
# PostProcessing
# create an instance of the MetricsPlotter class
metrics_plotter = MetricsPlotter(mse, val_mse, loss, val_loss)
metrics_plotter.plot(title='timeasfeature_CNNLSTM_past8andNext24DNN')
yhat1 = model_cnn_lstm.predict(X1, verbose=0)
num_cols = n_steps_out
# Generate column names
pred_cols = ['pt' + str(i + 1) for i in range(num_cols)]
real_cols = ['t' + str(i + 1) for i in range(num_cols)]
# Create dataframes
df_prediction = pd.DataFrame(yhat1, columns=pred_cols)
df_realvalues = pd.DataFrame(y1, columns=real_cols)
pred_1 = df_prediction['pt1']
Real_1 = df_realvalues['t1']
R2_for_first_hour.append(R_squared(Real_1, pred_1))
metric.add(y1, yhat1)
metrics_df = metric.get()
end_time = time.time()
times.append(end_time - start_time)

# RNN LSTM
start_time = time.time()
tf.keras.backend.clear_session()
# Define model architecture
hybrid_rnn_lstm_model = Sequential()
hybrid_rnn_lstm_model.add(
    Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps_in, n_features)))
hybrid_rnn_lstm_model.add(MaxPooling1D(pool_size=2))
hybrid_rnn_lstm_model.add(LSTM(64, activation='relu', return_sequences=True))
hybrid_rnn_lstm_model.add(Dropout(0.2))
hybrid_rnn_lstm_model.add(LSTM(128, activation='relu', return_sequences=True))
hybrid_rnn_lstm_model.add(Dropout(0.2))
hybrid_rnn_lstm_model.add(SimpleRNN(64, activation='relu', return_sequences=True))
hybrid_rnn_lstm_model.add(Dropout(0.2))
hybrid_rnn_lstm_model.add(SimpleRNN(32, activation='relu'))
hybrid_rnn_lstm_model.add(Dense(n_steps_out, activation='relu'))

# Compile the model
loss = tf.keras.losses.Huber()
hybrid_rnn_lstm_model.compile(loss=loss, optimizer='adam', metrics=['mse'])

# Define callbacks
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=30, verbose=1, restore_best_weights=True)

# Print model summary
hybrid_rnn_lstm_model.summary()

# Fit the model
history = hybrid_rnn_lstm_model.fit(X, y, epochs=ep, verbose=1, validation_data=(X1, y1),
                                    callbacks=[lr_scheduler, early_stopping])

# Get the mse and loss values from the history object
mse = history.history['mse']
loss = history.history['loss']
val_mse = history.history['val_mse']
val_loss = history.history['val_loss']

# Save the model
hybrid_rnn_lstm_model.save('RNNLSTM4.h5')

# PostProcessing
# create an instance of the MetricsPlotter class
metrics_plotter = MetricsPlotter(mse, val_mse, loss, val_loss)
metrics_plotter.plot(title='RNN LSTM')
yhat1 = hybrid_rnn_lstm_model.predict(X1, verbose=0)
num_cols = n_steps_out
# Generate column names
pred_cols = ['pt' + str(i + 1) for i in range(num_cols)]
real_cols = ['t' + str(i + 1) for i in range(num_cols)]
# Create dataframes
df_prediction = pd.DataFrame(yhat1, columns=pred_cols)
df_realvalues = pd.DataFrame(y1, columns=real_cols)
pred_1 = df_prediction['pt1']
Real_1 = df_realvalues['t1']
R2_for_first_hour.append(R_squared(Real_1, pred_1))
metric.add(y1, yhat1)
metrics_df = metric.get()
end_time = time.time()
times.append(end_time - start_time)

# CNN RNN
# clear previous session
start_time = time.time()
tf.keras.backend.clear_session()
model_cnn_rnn = Sequential()

# add a convolutional layer with 64 filters, a kernel size of 2, and ReLU activation
model_cnn_rnn.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps_in, n_features)))
# add a max pooling layer with a pool size of 2
model_cnn_rnn.add(MaxPooling1D(pool_size=2))
# add a dropout layer with a rate of 0.3
model_cnn_rnn.add(Dropout(0.3))

# add a SimpleRNN layer with 128 units and return sequences
model_cnn_rnn.add(SimpleRNN(128, activation='relu', return_sequences=True))
# add another dropout layer with a rate of 0.3
model_cnn_rnn.add(Dropout(0.3))

# add a flatten layer to convert the output of the previous layer to a 1D tensor
model_cnn_rnn.add(Flatten())
# add a dense layer with 50 units and ReLU activation
model_cnn_rnn.add(Dense(50, activation='relu'))
# add a dense layer with n_steps_out units and ReLU activation
model_cnn_rnn.add(Dense(n_steps_out, activation='relu'))

# define the loss function as Huber
loss = tf.keras.losses.Huber()
# compile the model with adam optimizer and mse as a metric
model_cnn_rnn.compile(loss=loss, optimizer='adam', metrics=['mse'])

# define the learning rate scheduler and early stopping callbacks
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=30, verbose=1, restore_best_weights=True)

# print a summary of the model
model_cnn_rnn.summary()

# fit the model
history = model_cnn_rnn.fit(X, y, epochs=ep, verbose=1, validation_data=(X1, y1),
                            callbacks=[lr_scheduler, early_stopping])

# save the model
model_cnn_rnn.save('CNNRNN4.h5')

# get the mse and loss values from the history object
mse = history.history['mse']
loss = history.history['loss']
val_mse = history.history['val_mse']
val_loss = history.history['val_loss']
# PostProcessing
# create an instance of the MetricsPlotter class
metrics_plotter = MetricsPlotter(mse, val_mse, loss, val_loss)
metrics_plotter.plot(title='CNNRNN')
yhat1 = model_cnn_rnn.predict(X1, verbose=0)
num_cols = n_steps_out
# Generate column names
pred_cols = ['pt' + str(i + 1) for i in range(num_cols)]
real_cols = ['t' + str(i + 1) for i in range(num_cols)]
# Create dataframes
df_prediction = pd.DataFrame(yhat1, columns=pred_cols)
df_realvalues = pd.DataFrame(y1, columns=real_cols)
pred_1 = df_prediction['pt1']
Real_1 = df_realvalues['t1']
R2_for_first_hour.append(R_squared(Real_1, pred_1))
metric.add(y1, yhat1)
metrics_df = metric.get()
end_time = time.time()
times.append(end_time - start_time)
# CNN
start_time = time.time()
tf.keras.backend.clear_session()
model_cnn = Sequential()
model_cnn.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps_in, n_features)))
model_cnn.add(MaxPooling1D(pool_size=2))
model_cnn.add(Flatten())
model_cnn.add(Dense(50, activation='relu'))
model_cnn.add(Dense(n_steps_out, activation='relu'))
loss = tf.keras.losses.Huber()
model_cnn.compile(loss=loss, optimizer='adam', metrics=['mse'])
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=30, verbose=1, restore_best_weights=True)
model_cnn.summary()
# fit model
history = model_cnn.fit(X, y, epochs=ep, verbose=1, validation_data=(X1, y1),
                        callbacks=[lr_scheduler, early_stopping])
# Get mae and loss from history log
mse = history.history['mse']
loss = history.history['loss']
# Get number of epochs
# get the mse and loss values from the history object
val_mse = history.history['val_mse']
val_loss = history.history['val_loss']
model_cnn.save('CNN4.h5')

# PostProcessing
# create an instance of the MetricsPlotter class
metrics_plotter = MetricsPlotter(mse, val_mse, loss, val_loss)
metrics_plotter.plot(title='CNN')
yhat1 = model_cnn.predict(X1, verbose=0)
num_cols = n_steps_out
# Generate column names
pred_cols = ['pt' + str(i + 1) for i in range(num_cols)]
real_cols = ['t' + str(i + 1) for i in range(num_cols)]
# Create dataframes
df_prediction = pd.DataFrame(yhat1, columns=pred_cols)
df_realvalues = pd.DataFrame(y1, columns=real_cols)
pred_1 = df_prediction['pt1']
Real_1 = df_realvalues['t1']
R2_for_first_hour.append(R_squared(Real_1, pred_1))
metric.add(y1, yhat1)
metrics_df = metric.get()
end_time = time.time()
times.append(end_time - start_time)

# RNN
start_time = time.time()
tf.keras.backend.clear_session()
model_rnn = Sequential()
model_rnn.add(SimpleRNN(100, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
model_rnn.add(SimpleRNN(100, activation='relu'))
model_rnn.add(Dense(n_steps_out, activation='relu'))
loss = tf.keras.losses.Huber()
model_rnn.compile(loss=loss, optimizer='adam', metrics=['mse'])
model_rnn.summary()
# fit model
history = model_rnn.fit(X, y, epochs=ep, verbose=1, validation_data=(X1, y1),
                        callbacks=[lr_scheduler, early_stopping])
# Get mae and loss from history log
mse = history.history['mse']
loss = history.history['loss']
# Get number of epochs
# get the mse and loss values from the history object
val_mse = history.history['val_mse']
val_loss = history.history['val_loss']
model_rnn.save('RNN4.h5')

# PostProcessing
# create an instance of the MetricsPlotter class
metrics_plotter = MetricsPlotter(mse, val_mse, loss, val_loss)
metrics_plotter.plot(title='RNN')
yhat1 = model_rnn.predict(X1, verbose=0)
num_cols = n_steps_out
# Generate column names
pred_cols = ['pt' + str(i + 1) for i in range(num_cols)]
real_cols = ['t' + str(i + 1) for i in range(num_cols)]
# Create dataframes
df_prediction = pd.DataFrame(yhat1, columns=pred_cols)
df_realvalues = pd.DataFrame(y1, columns=real_cols)
pred_1 = df_prediction['pt1']
Real_1 = df_realvalues['t1']
R2_for_first_hour.append(R_squared(Real_1, pred_1))
metric.add(y1, yhat1)
metrics_df = metric.get()
end_time = time.time()
times.append(end_time - start_time)
metrics_df.to_csv('metric4', index=False)

import csv

# Specify the CSV file path
csv_file = 'Times4.csv'

# Open the CSV file in write mode
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)

    # Iterate over the list and write each element as a row in the CSV file
    for item in times:
        writer.writerow([item])


s = List_S_fea[5]
for j in s:
    def split_data(series):
        """Splits input series into train, val and test.
        """
        train_data = series[:26280]  # 2011 start from 70080 done saved 11 , 61320 , 52560 , 43800
        test_data = series[26280:]
        return train_data, test_data


    vars()[f"in_seqt{j}"] = train_multi[j].values
    vars()[f"in_seqt{j}"].reshape((len(train_multi[j]), 1))
    vars()[f"in_seqt_test{j}"] = test_multi[j].values
    vars()[f"in_seqt_test{j}"].reshape((len(vars()[f"in_seqt_test{j}"]), 1))
out_seqt = in_seqtHeating
dataset = hstack((in_seqtDirectNormalRadiation.reshape(-1, 1), in_seqtDryBulbTemperature.reshape(-1, 1),
                  in_seqtOccupancySchedule.reshape(-1, 1), in_seqtdayofw.reshape(-1, 1),
                  in_seqtHeating.reshape(-1, 1), out_seqt.reshape(-1, 1)))
out_seqt_test = in_seqt_testHeating
dataset_test = hstack((in_seqt_testDirectNormalRadiation.reshape(-1, 1),
                       in_seqt_testDryBulbTemperature.reshape(-1, 1),
                       in_seqt_testOccupancySchedule.reshape(-1, 1), in_seqt_testdayofw.reshape(-1, 1),
                       in_seqt_testHeating.reshape(-1, 1), out_seqt_test.reshape(-1, 1)))
n_steps_in, n_steps_out = 16, 24
X, y = split_sequences(dataset, n_steps_in, n_steps_out)
X1, y1 = split_sequences(dataset_test, n_steps_in, n_steps_out)
n_features = X.shape[2]

# LSTM
start_time = time.time()
tf.keras.backend.clear_session()
model = Sequential()
model.add(LSTM(128, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
model.add(Dropout(0.3))
model.add(LSTM(128, activation='relu', return_sequences=True))
model.add(Dropout(0.3))
# model.add(LSTM(128, activation='relu', return_sequences=True))
# model.add(Dropout(0.3))
model.add(LSTM(128, activation='relu'))
model.add(Dense(n_steps_out, activation='relu'))
# set up the learning rate scheduler and early stopping
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=30, verbose=1, restore_best_weights=True)
# compile the model with the Adam optimizer and Huber loss
model.compile(optimizer=Adam(lr=1e-4), loss=tf.keras.losses.Huber(), metrics=['mse'])
# train the model with early stopping and the learning rate scheduler
history = model.fit(X, y, epochs=ep, verbose=1, validation_data=(X1, y1),
                    callbacks=[lr_scheduler, early_stopping])
# Get mae and loss from history log
mse = history.history['mse']
loss = history.history['loss']
# Get number of epochs
# get the mse and loss values from the history object
val_mse = history.history['val_mse']
val_loss = history.history['val_loss']
# save model
model.save('LSTM5.h5')

# PostProcessing
# create an instance of the MetricsPlotter class
metrics_plotter = MetricsPlotter(mse, val_mse, loss, val_loss)
metrics_plotter.plot(title='LSTM')
yhat1 = model.predict(X1, verbose=0)
num_cols = n_steps_out
# Generate column names
pred_cols = ['pt' + str(i + 1) for i in range(num_cols)]
real_cols = ['t' + str(i + 1) for i in range(num_cols)]
# Create dataframes
df_prediction = pd.DataFrame(yhat1, columns=pred_cols)
df_realvalues = pd.DataFrame(y1, columns=real_cols)
pred_1 = df_prediction['pt1']
Real_1 = df_realvalues['t1']
R2_for_first_hour.append(R_squared(Real_1, pred_1))
metric.add(y1, yhat1)
metrics_df = metric.get()
end_time = time.time()
times.append(end_time - start_time)

# CNN_lstm

# clear previous session
start_time = time.time()
tf.keras.backend.clear_session()
model_cnn_lstm = Sequential()
# add a convolutional layer with 64 filters, a kernel size of 2, and ReLU activation
model_cnn_lstm.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps_in, n_features)))
# add a max pooling layer with a pool size of 2
model_cnn_lstm.add(MaxPooling1D(pool_size=2))
# add a dropout layer with a rate of 0.3
model_cnn_lstm.add(Dropout(0.3))

# add an LSTM layer with 128 units and return sequences
model_cnn_lstm.add(LSTM(128, activation='relu', return_sequences=True))
# add another dropout layer with a rate of 0.3
model_cnn_lstm.add(Dropout(0.3))

# add a flatten layer to convert the output of the previous layer to a 1D tensor
model_cnn_lstm.add(Flatten())
# add a dense layer with 50 units and ReLU activation
model_cnn_lstm.add(Dense(50, activation='relu'))
# add a dense layer with n_steps_out units and ReLU activation
model_cnn_lstm.add(Dense(n_steps_out, activation='relu'))

# define the loss function as Huber
loss = tf.keras.losses.Huber()
# compile the model with adam optimizer and mse as a metric
model_cnn_lstm.compile(loss=loss, optimizer='adam', metrics=['mse'])

# define the learning rate scheduler and early stopping callbacks
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=30, verbose=1, restore_best_weights=True)

# print a summary of the model
model_cnn_lstm.summary()

# fit the model
history = model_cnn_lstm.fit(X, y, epochs=ep, verbose=1, validation_data=(X1, y1),
                             callbacks=[lr_scheduler, early_stopping])

# save the model
model_cnn_lstm.save('CNNLSTM5.h5')

# get the mse and loss values from the history object
mse = history.history['mse']
loss = history.history['loss']
val_mse = history.history['val_mse']
val_loss = history.history['val_loss']
# PostProcessing
# create an instance of the MetricsPlotter class
metrics_plotter = MetricsPlotter(mse, val_mse, loss, val_loss)
metrics_plotter.plot(title='timeasfeature_CNNLSTM_past8andNext24DNN')
yhat1 = model_cnn_lstm.predict(X1, verbose=0)
num_cols = n_steps_out
# Generate column names
pred_cols = ['pt' + str(i + 1) for i in range(num_cols)]
real_cols = ['t' + str(i + 1) for i in range(num_cols)]
# Create dataframes
df_prediction = pd.DataFrame(yhat1, columns=pred_cols)
df_realvalues = pd.DataFrame(y1, columns=real_cols)
pred_1 = df_prediction['pt1']
Real_1 = df_realvalues['t1']
R2_for_first_hour.append(R_squared(Real_1, pred_1))
metric.add(y1, yhat1)
metrics_df = metric.get()
end_time = time.time()
times.append(end_time - start_time)

# RNN LSTM
start_time = time.time()
tf.keras.backend.clear_session()
# Define model architecture
hybrid_rnn_lstm_model = Sequential()
hybrid_rnn_lstm_model.add(
    Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps_in, n_features)))
hybrid_rnn_lstm_model.add(MaxPooling1D(pool_size=2))
hybrid_rnn_lstm_model.add(LSTM(64, activation='relu', return_sequences=True))
hybrid_rnn_lstm_model.add(Dropout(0.2))
hybrid_rnn_lstm_model.add(LSTM(128, activation='relu', return_sequences=True))
hybrid_rnn_lstm_model.add(Dropout(0.2))
hybrid_rnn_lstm_model.add(SimpleRNN(64, activation='relu', return_sequences=True))
hybrid_rnn_lstm_model.add(Dropout(0.2))
hybrid_rnn_lstm_model.add(SimpleRNN(32, activation='relu'))
hybrid_rnn_lstm_model.add(Dense(n_steps_out, activation='relu'))

# Compile the model
loss = tf.keras.losses.Huber()
hybrid_rnn_lstm_model.compile(loss=loss, optimizer='adam', metrics=['mse'])

# Define callbacks
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=30, verbose=1, restore_best_weights=True)

# Print model summary
hybrid_rnn_lstm_model.summary()

# Fit the model
history = hybrid_rnn_lstm_model.fit(X, y, epochs=ep, verbose=1, validation_data=(X1, y1),
                                    callbacks=[lr_scheduler, early_stopping])

# Get the mse and loss values from the history object
mse = history.history['mse']
loss = history.history['loss']
val_mse = history.history['val_mse']
val_loss = history.history['val_loss']

# Save the model
hybrid_rnn_lstm_model.save('RNNLSTM5.h5')

# PostProcessing
# create an instance of the MetricsPlotter class
metrics_plotter = MetricsPlotter(mse, val_mse, loss, val_loss)
metrics_plotter.plot(title='RNN LSTM')
yhat1 = hybrid_rnn_lstm_model.predict(X1, verbose=0)
num_cols = n_steps_out
# Generate column names
pred_cols = ['pt' + str(i + 1) for i in range(num_cols)]
real_cols = ['t' + str(i + 1) for i in range(num_cols)]
# Create dataframes
df_prediction = pd.DataFrame(yhat1, columns=pred_cols)
df_realvalues = pd.DataFrame(y1, columns=real_cols)
pred_1 = df_prediction['pt1']
Real_1 = df_realvalues['t1']
R2_for_first_hour.append(R_squared(Real_1, pred_1))
metric.add(y1, yhat1)
metrics_df = metric.get()
end_time = time.time()
times.append(end_time - start_time)

# CNN RNN
# clear previous session
start_time = time.time()
tf.keras.backend.clear_session()
model_cnn_rnn = Sequential()

# add a convolutional layer with 64 filters, a kernel size of 2, and ReLU activation
model_cnn_rnn.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps_in, n_features)))
# add a max pooling layer with a pool size of 2
model_cnn_rnn.add(MaxPooling1D(pool_size=2))
# add a dropout layer with a rate of 0.3
model_cnn_rnn.add(Dropout(0.3))

# add a SimpleRNN layer with 128 units and return sequences
model_cnn_rnn.add(SimpleRNN(128, activation='relu', return_sequences=True))
# add another dropout layer with a rate of 0.3
model_cnn_rnn.add(Dropout(0.3))

# add a flatten layer to convert the output of the previous layer to a 1D tensor
model_cnn_rnn.add(Flatten())
# add a dense layer with 50 units and ReLU activation
model_cnn_rnn.add(Dense(50, activation='relu'))
# add a dense layer with n_steps_out units and ReLU activation
model_cnn_rnn.add(Dense(n_steps_out, activation='relu'))

# define the loss function as Huber
loss = tf.keras.losses.Huber()
# compile the model with adam optimizer and mse as a metric
model_cnn_rnn.compile(loss=loss, optimizer='adam', metrics=['mse'])

# define the learning rate scheduler and early stopping callbacks
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=30, verbose=1, restore_best_weights=True)

# print a summary of the model
model_cnn_rnn.summary()

# fit the model
history = model_cnn_rnn.fit(X, y, epochs=ep, verbose=1, validation_data=(X1, y1),
                            callbacks=[lr_scheduler, early_stopping])

# save the model
model_cnn_rnn.save('CNNRNN5.h5')

# get the mse and loss values from the history object
mse = history.history['mse']
loss = history.history['loss']
val_mse = history.history['val_mse']
val_loss = history.history['val_loss']
# PostProcessing
# create an instance of the MetricsPlotter class
metrics_plotter = MetricsPlotter(mse, val_mse, loss, val_loss)
metrics_plotter.plot(title='CNNRNN')
yhat1 = model_cnn_rnn.predict(X1, verbose=0)
num_cols = n_steps_out
# Generate column names
pred_cols = ['pt' + str(i + 1) for i in range(num_cols)]
real_cols = ['t' + str(i + 1) for i in range(num_cols)]
# Create dataframes
df_prediction = pd.DataFrame(yhat1, columns=pred_cols)
df_realvalues = pd.DataFrame(y1, columns=real_cols)
pred_1 = df_prediction['pt1']
Real_1 = df_realvalues['t1']
R2_for_first_hour.append(R_squared(Real_1, pred_1))
metric.add(y1, yhat1)
metrics_df = metric.get()
end_time = time.time()
times.append(end_time - start_time)
# CNN
start_time = time.time()
tf.keras.backend.clear_session()
model_cnn = Sequential()
model_cnn.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps_in, n_features)))
model_cnn.add(MaxPooling1D(pool_size=2))
model_cnn.add(Flatten())
model_cnn.add(Dense(50, activation='relu'))
model_cnn.add(Dense(n_steps_out, activation='relu'))
loss = tf.keras.losses.Huber()
model_cnn.compile(loss=loss, optimizer='adam', metrics=['mse'])
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=30, verbose=1, restore_best_weights=True)
model_cnn.summary()
# fit model
history = model_cnn.fit(X, y, epochs=ep, verbose=1, validation_data=(X1, y1),
                        callbacks=[lr_scheduler, early_stopping])
# Get mae and loss from history log
mse = history.history['mse']
loss = history.history['loss']
# Get number of epochs
# get the mse and loss values from the history object
val_mse = history.history['val_mse']
val_loss = history.history['val_loss']
model_cnn.save('CNN5.h5')

# PostProcessing
# create an instance of the MetricsPlotter class
metrics_plotter = MetricsPlotter(mse, val_mse, loss, val_loss)
metrics_plotter.plot(title='CNN')
yhat1 = model_cnn.predict(X1, verbose=0)
num_cols = n_steps_out
# Generate column names
pred_cols = ['pt' + str(i + 1) for i in range(num_cols)]
real_cols = ['t' + str(i + 1) for i in range(num_cols)]
# Create dataframes
df_prediction = pd.DataFrame(yhat1, columns=pred_cols)
df_realvalues = pd.DataFrame(y1, columns=real_cols)
pred_1 = df_prediction['pt1']
Real_1 = df_realvalues['t1']
R2_for_first_hour.append(R_squared(Real_1, pred_1))
metric.add(y1, yhat1)
metrics_df = metric.get()
end_time = time.time()
times.append(end_time - start_time)

# RNN
start_time = time.time()
tf.keras.backend.clear_session()
model_rnn = Sequential()
model_rnn.add(SimpleRNN(100, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
model_rnn.add(SimpleRNN(100, activation='relu'))
model_rnn.add(Dense(n_steps_out, activation='relu'))
loss = tf.keras.losses.Huber()
model_rnn.compile(loss=loss, optimizer='adam', metrics=['mse'])
model_rnn.summary()
# fit model
history = model_rnn.fit(X, y, epochs=ep, verbose=1, validation_data=(X1, y1),
                        callbacks=[lr_scheduler, early_stopping])
# Get mae and loss from history log
mse = history.history['mse']
loss = history.history['loss']
# Get number of epochs
# get the mse and loss values from the history object
val_mse = history.history['val_mse']
val_loss = history.history['val_loss']
model_rnn.save('RNN5.h5')

# PostProcessing
# create an instance of the MetricsPlotter class
metrics_plotter = MetricsPlotter(mse, val_mse, loss, val_loss)
metrics_plotter.plot(title='RNN')
yhat1 = model_rnn.predict(X1, verbose=0)
num_cols = n_steps_out
# Generate column names
pred_cols = ['pt' + str(i + 1) for i in range(num_cols)]
real_cols = ['t' + str(i + 1) for i in range(num_cols)]
# Create dataframes
df_prediction = pd.DataFrame(yhat1, columns=pred_cols)
df_realvalues = pd.DataFrame(y1, columns=real_cols)
pred_1 = df_prediction['pt1']
Real_1 = df_realvalues['t1']
R2_for_first_hour.append(R_squared(Real_1, pred_1))
metric.add(y1, yhat1)
metrics_df = metric.get()
end_time = time.time()
times.append(end_time - start_time)
metrics_df.to_csv('metric5', index=False)

import csv

# Specify the CSV file path
csv_file = 'Times5.csv'

# Open the CSV file in write mode
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)

    # Iterate over the list and write each element as a row in the CSV file
    for item in times:
        writer.writerow([item])


