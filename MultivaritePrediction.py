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
multivar_df_time =load_data(imput_col_time, verbose = True )
# data with no time as feature

# multivar_df_time = normalize_data(multivar_df_time)



imput_col = ['DryBulbTemperature','DirectNormalRadiation','GlobalHorizontalRadiation','SnowDepth','OccupancySchedule',
                         'WindDirection','Heating']
multivar_df = load_data(imput_col, verbose = True )
# data based on papers and my experties
imput_col_manual = ['DryBulbTemperature','DirectNormalRadiation','OccupancySchedule'
                         ,'dayofw','Heating']


multivar_df_manual = load_data(imput_col_manual, verbose = True )

new_names1 = {col: col.replace(' ', '') for col in multivar_df_time.columns}
multivar_df_time = multivar_df_time.rename(columns=new_names1)

new_names2 = {col: col.replace(' ', '') for col in multivar_df.columns}
multivar_df = multivar_df.rename(columns=new_names2)

new_names3 = {col: col.replace(' ', '') for col in multivar_df_manual.columns}
multivar_df_manual = multivar_df_manual.rename(columns=new_names3)

List_S_fea = [multivar_df_time, multivar_df, multivar_df_manual ]
input_features = [imput_col_time,imput_col,imput_col_manual]
List_S_fea[0]['dayofw'] = List_S_fea[0]['dayofw'].astype(float)

# using normalized data
# using trigonometry functions
R2_for_first_hour = []
times = []
for i in range(len(input_features)):
    start_time = time.time()
    s = List_S_fea[i]
    train_multi, test_multi = split_data(s)
    ep = 300
    metric = Metric()
    if i == 0:
        for j in s:
            vars()[f"in_seqt{j}"] = train_multi[j].values
            vars()[f"in_seqt{j}"].reshape((len(train_multi[j]), 1))
            vars()[f"in_seqt_test{j}"] = test_multi[j].values
            vars()[f"in_seqt_test{j}"].reshape((len(vars()[f"in_seqt_test{j}"]), 1))
        out_seqt = in_seqtHeating
        dataset = hstack((in_seqtDirectNormalRadiation.reshape(-1, 1) ,in_seqtDryBulbTemperature.reshape(-1, 1),
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
# DNN
        tf.keras.backend.clear_session()
        model_dnn = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(n_steps_in, n_features)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(n_steps_out, activation='relu')
        ], name='dnn')
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
        loss = tf.keras.losses.Huber()
        model_dnn.compile(loss=loss, optimizer='adam', metrics=['mse'])
        model_dnn.summary()
        # fit the model
        history = model_dnn.fit(X, y, epochs=ep, verbose=1, validation_data=(X1, y1),
                                callbacks=[lr_scheduler, early_stopping])
        # Create a tf.data.Dataset from the input data
        batch_size = 32
        shuffle_buffer_size = 1000
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        dataset = dataset.shuffle(shuffle_buffer_size).batch(batch_size)
        # Get mae and loss from history log
        mse = history.history['mse']
        loss = history.history['loss']
        # Get number of epochs
        # get the mse and loss values from the history object
        val_mse = history.history['val_mse']
        val_loss = history.history['val_loss']
        model_dnn.save('timeasfeature_DDN_past8andNext24DNN.h5')

        # PostProcessing
        # create an instance of the MetricsPlotter class
        metrics_plotter = MetricsPlotter(mse, val_mse, loss, val_loss)
        metrics_plotter.plot(title='timeasfeature_DDN_past8andNext24DNN')
        yhat1 = model_dnn.predict(X1, verbose=0)
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

        # set the x-axis label and tick marks
        plt.xlabel('Epoch')
        plt.xticks(range(len(mse)), range(1, len(mse) + 1))
        plt.show()
        end_time = time.time()
        times.append(end_time - start_time)


        # LSTM
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
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
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
        model.save('timeasfeature_lstm_past8andNext24.h5')

# PostProcessing
        # create an instance of the MetricsPlotter class
        metrics_plotter = MetricsPlotter(mse, val_mse, loss, val_loss)
        metrics_plotter.plot(title='timeasfeature_lstm_past8andNext24')
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
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

        # print a summary of the model
        model_cnn_lstm.summary()

        # fit the model
        history = model_cnn_lstm.fit(X, y, epochs=ep, verbose=1, validation_data=(X1, y1),
                                     callbacks=[lr_scheduler, early_stopping])

        # save the model
        model_cnn_lstm.save('timeasfeature_CNNLSTM_past8andNext24DNN.h5')

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
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

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
        hybrid_rnn_lstm_model.save('timeasfeature_RNNLSTM_past8andNext24DNN.h5')

# PostProcessing
        # create an instance of the MetricsPlotter class
        metrics_plotter = MetricsPlotter(mse, val_mse, loss, val_loss)
        metrics_plotter.plot(title='timeasfeature_RNNLSTM_past8andNext24DNN')
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

        # CNN RNN LSTM
        model_cnn_lstm_rnn = Sequential()

        # add a convolutional layer with 64 filters, a kernel size of 2, and ReLU activation
        model_cnn_lstm_rnn.add(
            Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps_in, n_features)))
        # add a max pooling layer with a pool size of 2
        model_cnn_lstm_rnn.add(MaxPooling1D(pool_size=2))
        # add a dropout layer with a rate of 0.3
        model_cnn_lstm_rnn.add(Dropout(0.3))

        # add an RNN layer with 128 units and return sequences
        model_cnn_lstm_rnn.add(SimpleRNN(128, activation='relu', return_sequences=True))
        # add an LSTM layer with 256 units and return sequences
        model_cnn_lstm_rnn.add(LSTM(256, activation='relu', return_sequences=True))
        # add another dropout layer with a rate of 0.3
        model_cnn_lstm_rnn.add(Dropout(0.3))

        # add a flatten layer to convert the output of the previous layer to a 1D tensor
        model_cnn_lstm_rnn.add(Flatten())
        # add a dense layer with 50 units and ReLU activation
        model_cnn_lstm_rnn.add(Dense(50, activation='relu'))
        # add a dense layer with n_steps_out units and ReLU activation
        model_cnn_lstm_rnn.add(Dense(n_steps_out, activation='relu'))

        # define the loss function as Huber
        loss = tf.keras.losses.Huber()
        # compile the model with adam optimizer and mse as a metric
        model_cnn_lstm_rnn.compile(loss=loss, optimizer='adam', metrics=['mse'])

        # define the learning rate scheduler and early stopping callbacks
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

        # print a summary of the model
        model_cnn_lstm_rnn.summary()

        # fit the model
        history = model_cnn_lstm_rnn.fit(X, y, epochs=ep, verbose=1, validation_data=(X1, y1),
                                         callbacks=[lr_scheduler, early_stopping])

        # save the model
        model_cnn_lstm_rnn.save('timeasfeature_CNNRNNLSTM_past8andNext24DNN.h5')

        # get the mse and loss values from the history object
        mse = history.history['mse']
        loss = history.history['loss']
        val_mse = history.history['val_mse']
        val_loss = history.history['val_loss']

        # PostProcessing
        # create an instance of the MetricsPlotter class
        metrics_plotter = MetricsPlotter(mse, val_mse, loss, val_loss)
        metrics_plotter.plot(title='timeasfeature_CNNRNNLSTM_past8andNext24DNN')
        yhat1 = model_cnn_lstm_rnn.predict(X1, verbose=0)
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
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
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
        model_cnn.save('timeasfeature_CNN_past8andNext24DNN.h5')

# PostProcessing
        # create an instance of the MetricsPlotter class
        metrics_plotter = MetricsPlotter(mse, val_mse, loss, val_loss)
        metrics_plotter.plot(title='timeasfeature_CNN_past8andNext24DNN')
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
        model_rnn.save('timeasfeature_rnn_past8andNext24DNN.h5')

# PostProcessing
        # create an instance of the MetricsPlotter class
        metrics_plotter = MetricsPlotter(mse, val_mse, loss, val_loss)
        metrics_plotter.plot(title='timeasfeature_rnn_past8andNext24DNN')
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
'''
    elif i == 1:
        for j in s:
            vars()[f"in_seq{j}"] = train_multi[j].values
            vars()[f"in_seq{j}"].reshape((len(train_multi[j]), 1))
            vars()[f"in_seq_test{j}"] = test_multi[j].values
            vars()[f"in_seq_test{j}"].reshape((len(vars()[f"in_seq_test{j}"]), 1))

        out_seq = in_seqHeating
        dataset = hstack((in_seqDirectNormalRadiation.reshape(-1, 1), in_seqDryBulbTemperature.reshape(-1, 1),
                          in_seqOccupancySchedule.reshape(-1, 1), in_seqSnowDepth.reshape(-1, 1),
                          in_seqGlobalHorizontalRadiation.reshape(-1, 1)
                          , in_seqWindDirection.reshape(-1, 1), in_seqHeating.reshape(-1, 1), out_seq.reshape(-1, 1)))
        out_seq_test = in_seq_testHeating
        dataset_test = hstack((in_seq_testDirectNormalRadiation.reshape(-1, 1),
                               in_seq_testDryBulbTemperature.reshape(-1, 1),
                               in_seq_testOccupancySchedule.reshape(-1, 1), in_seq_testSnowDepth.reshape(-1, 1),
                               in_seq_testGlobalHorizontalRadiation.reshape(-1, 1)
                               , in_seq_testWindDirection.reshape(-1, 1), in_seq_testHeating.reshape(-1, 1),
                               out_seq_test.reshape(-1, 1)))
        n_steps_in, n_steps_out = 8, 24
        X, y = split_sequences(dataset, n_steps_in, n_steps_out)
        X1, y1 = split_sequences(dataset_test, n_steps_in, n_steps_out)
        n_features = X.shape[2]

        # LSTM
        model = Sequential()
        model.add(LSTM(128, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
        model.add(Dropout(0.3))
        model.add(LSTM(256, activation='relu', return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(256, activation='relu', return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(128, activation='relu'))
        model.add(Dense(n_steps_out, activation='relu'))
        # set up the learning rate scheduler and early stopping
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
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
        model.save('lstm_past8andNext24.h5')
        # create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # plot the mse values on the first subplot
        ax1.plot(mse, label='Training MSE')
        ax1.plot(val_mse, label='Validation MSE')
        ax1.set_ylabel('MSE')
        ax1.legend()

        # plot the loss values on the second subplot
        ax2.plot(loss, label='Training Loss')
        ax2.plot(val_loss, label='Validation Loss')
        ax2.set_ylabel('Loss')
        ax2.legend()

        # set the x-axis label and tick marks
        plt.xlabel('Epoch')
        plt.xticks(range(len(mse)), range(1, len(mse) + 1))
        plt.show()
        end_time = time.time()
        times.append(end_time - start_time)

        # DNN
        tf.keras.backend.clear_session()
        model_dnn = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(n_steps_in, n_features)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(n_steps_out, activation='relu')
        ], name='dnn')
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
        loss = tf.keras.losses.Huber()
        model_dnn.compile(loss=loss, optimizer='adam', metrics=['mse'])
        model_dnn.summary()
        # fit the model
        history = model_dnn.fit(X, y, epochs=ep, verbose=1, validation_data=(X1, y1),
                                callbacks=[lr_scheduler, early_stopping])
        # Get mae and loss from history log
        mse = history.history['mse']
        loss = history.history['loss']
        # Get number of epochs
        # get the mse and loss values from the history object
        val_mse = history.history['val_mse']
        val_loss = history.history['val_loss']
        model_dnn.save('DDN_past8andNext24DNN.h5')
        # create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # plot the mse values on the first subplot
        ax1.plot(mse, label='Training MSE')
        ax1.plot(val_mse, label='Validation MSE')
        ax1.set_ylabel('MSE')
        ax1.legend()

        # plot the loss values on the second subplot
        ax2.plot(loss, label='Training Loss')
        ax2.plot(val_loss, label='Validation Loss')
        ax2.set_ylabel('Loss')
        ax2.legend()

        # set the x-axis label and tick marks
        plt.xlabel('Epoch')
        plt.xticks(range(len(mse)), range(1, len(mse) + 1))
        plt.show()
        end_time = time.time()
        times.append(end_time - start_time)

        # CNN
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
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
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
        model_cnn.save('CNN_past8andNext24DNN.h5')
        # create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # plot the mse values on the first subplot
        ax1.plot(mse, label='Training MSE')
        ax1.plot(val_mse, label='Validation MSE')
        ax1.set_ylabel('MSE')
        ax1.legend()

        # plot the loss values on the second subplot
        ax2.plot(loss, label='Training Loss')
        ax2.plot(val_loss, label='Validation Loss')
        ax2.set_ylabel('Loss')
        ax2.legend()

        # set the x-axis label and tick marks
        plt.xlabel('Epoch')
        plt.xticks(range(len(mse)), range(1, len(mse) + 1))
        plt.show()
        end_time = time.time()
        times.append(end_time - start_time)

        # RNN
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
        model_rnn.save('rnn_past8andNext24DNN.h5')
        # create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # plot the mse values on the first subplot
        ax1.plot(mse, label='Training MSE')
        ax1.plot(val_mse, label='Validation MSE')
        ax1.set_ylabel('MSE')
        ax1.legend()

        # plot the loss values on the second subplot
        ax2.plot(loss, label='Training Loss')
        ax2.plot(val_loss, label='Validation Loss')
        ax2.set_ylabel('Loss')
        ax2.legend()

        # set the x-axis label and tick marks
        plt.xlabel('Epoch')
        plt.xticks(range(len(mse)), range(1, len(mse) + 1))
        plt.show()
        end_time = time.time()
        times.append(end_time - start_time)

        # CNN_lstm
        # clear previous session
        # tf.keras.backend.clear_session()
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
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

        # print a summary of the model
        model_cnn_lstm.summary()

        # fit the model
        history = model_cnn_lstm.fit(X, y, epochs=ep, verbose=1, validation_data=(X1, y1),
                                     callbacks=[lr_scheduler, early_stopping])

        # save the model
        model_cnn_lstm.save('CNNLSTM_past8andNext24DNN.h5')

        # get the mse and loss values from the history object
        mse = history.history['mse']
        loss = history.history['loss']
        val_mse = history.history['val_mse']
        val_loss = history.history['val_loss']
        # create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # plot the mse values on the first subplot
        ax1.plot(mse, label='Training MSE')
        ax1.plot(val_mse, label='Validation MSE')
        ax1.set_ylabel('MSE')
        ax1.legend()

        # plot the loss values on the second subplot
        ax2.plot(loss, label='Training Loss')
        ax2.plot(val_loss, label='Validation Loss')
        ax2.set_ylabel('Loss')
        ax2.legend()

        # set the x-axis label and tick marks
        plt.xlabel('Epoch')
        plt.xticks(range(len(mse)), range(1, len(mse) + 1))
        plt.show()
        end_time = time.time()
        times.append(end_time - start_time)

        # RNN LSTM
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
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

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
        # create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # plot the mse values on the first subplot
        ax1.plot(mse, label='Training MSE')
        ax1.plot(val_mse, label='Validation MSE')
        ax1.set_ylabel('MSE')
        ax1.legend()

        # plot the loss values on the second subplot
        ax2.plot(loss, label='Training Loss')
        ax2.plot(val_loss, label='Validation Loss')
        ax2.set_ylabel('Loss')
        ax2.legend()

        # set the x-axis label and tick marks
        plt.xlabel('Epoch')
        plt.xticks(range(len(mse)), range(1, len(mse) + 1))
        plt.show()
        # Save the model
        hybrid_rnn_lstm_model.save('RNNLSTM_past8andNext24DNN.h5')
        end_time = time.time()
        times.append(end_time - start_time)

        # CNN RNN LSTM
        model_cnn_lstm_rnn = Sequential()

        # add a convolutional layer with 64 filters, a kernel size of 2, and ReLU activation
        model_cnn_lstm_rnn.add(
            Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps_in, n_features)))
        # add a max pooling layer with a pool size of 2
        model_cnn_lstm_rnn.add(MaxPooling1D(pool_size=2))
        # add a dropout layer with a rate of 0.3
        model_cnn_lstm_rnn.add(Dropout(0.3))

        # add an RNN layer with 128 units and return sequences
        model_cnn_lstm_rnn.add(SimpleRNN(128, activation='relu', return_sequences=True))
        # add an LSTM layer with 256 units and return sequences
        model_cnn_lstm_rnn.add(LSTM(256, activation='relu', return_sequences=True))
        # add another dropout layer with a rate of 0.3
        model_cnn_lstm_rnn.add(Dropout(0.3))

        # add a flatten layer to convert the output of the previous layer to a 1D tensor
        model_cnn_lstm_rnn.add(Flatten())
        # add a dense layer with 50 units and ReLU activation
        model_cnn_lstm_rnn.add(Dense(50, activation='relu'))
        # add a dense layer with n_steps_out units and ReLU activation
        model_cnn_lstm_rnn.add(Dense(n_steps_out, activation='relu'))

        # define the loss function as Huber
        loss = tf.keras.losses.Huber()
        # compile the model with adam optimizer and mse as a metric
        model_cnn_lstm_rnn.compile(loss=loss, optimizer='adam', metrics=['mse'])

        # define the learning rate scheduler and early stopping callbacks
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

        # print a summary of the model
        model_cnn_lstm_rnn.summary()

        # fit the model
        history = model_cnn_lstm_rnn.fit(X, y, epochs=ep, verbose=1, validation_data=(X1, y1),
                                         callbacks=[lr_scheduler, early_stopping])

        # save the model
        model_cnn_lstm_rnn.save('CNNRNNLSTM_past8andNext24DNN.h5')

        # get the mse and loss values from the history object
        mse = history.history['mse']
        loss = history.history['loss']
        val_mse = history.history['val_mse']
        val_loss = history.history['val_loss']
        # create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # plot the mse values on the first subplot
        ax1.plot(mse, label='Training MSE')
        ax1.plot(val_mse, label='Validation MSE')
        ax1.set_ylabel('MSE')
        ax1.legend()

        # plot the loss values on the second subplot
        ax2.plot(loss, label='Training Loss')
        ax2.plot(val_loss, label='Validation Loss')
        ax2.set_ylabel('Loss')
        ax2.legend()

        # set the x-axis label and tick marks
        plt.xlabel('Epoch')
        plt.xticks(range(len(mse)), range(1, len(mse) + 1))
        plt.show()
        end_time = time.time()
        times.append(end_time - start_time)
'''
'''
    elif i == 2:
        for j in s:
            vars()[f"in_seq{j}"] = train_multi[j].values
            vars()[f"in_seq{j}"].reshape((len(train_multi[j]), 1))
            vars()[f"in_seq_test{j}"] = test_multi[j].values
            vars()[f"in_seq_test{j}"].reshape((len(vars()[f"in_seq_test{j}"]), 1))
        out_seq = in_seqHeating
        dataset = hstack((in_seqDryBulbTemperature.reshape(-1, 1), in_seqOccupancySchedule.reshape(-1, 1)
                          , in_seqHeating.reshape(-1, 1), out_seq.reshape(-1, 1)))
        out_seq_test = in_seq_testHeating
        dataset_test = hstack((in_seq_testDryBulbTemperature.reshape(-1, 1), in_seq_testOccupancySchedule.reshape(-1, 1)
                               , in_seq_testHeating.reshape(-1, 1), out_seq_test.reshape(-1, 1)))
        n_steps_in, n_steps_out = 8, 24
        X, y = split_sequences(dataset, n_steps_in, n_steps_out)
        X1, y1 = split_sequences(dataset_test, n_steps_in, n_steps_out)
        n_features = X.shape[2]

        # LSTM
        model = Sequential()
        model.add(LSTM(128, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
        model.add(Dropout(0.3))
        model.add(LSTM(256, activation='relu', return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(256, activation='relu', return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(128, activation='relu'))
        model.add(Dense(n_steps_out, activation='relu'))
        # set up the learning rate scheduler and early stopping
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
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
        model.save('M_lstm_past8andNext24.h5')
        # create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # plot the mse values on the first subplot
        ax1.plot(mse, label='Training MSE')
        ax1.plot(val_mse, label='Validation MSE')
        ax1.set_ylabel('MSE')
        ax1.legend()

        # plot the loss values on the second subplot
        ax2.plot(loss, label='Training Loss')
        ax2.plot(val_loss, label='Validation Loss')
        ax2.set_ylabel('Loss')
        ax2.legend()

        # set the x-axis label and tick marks
        plt.xlabel('Epoch')
        plt.xticks(range(len(mse)), range(1, len(mse) + 1))
        plt.show()
        end_time = time.time()
        times.append(end_time - start_time)

        # DNN
        tf.keras.backend.clear_session()
        model_dnn = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(n_steps_in, n_features)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(n_steps_out, activation='relu')
        ], name='dnn')
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
        loss = tf.keras.losses.Huber()
        model_dnn.compile(loss=loss, optimizer='adam', metrics=['mse'])
        model_dnn.summary()
        # fit the model
        history = model_dnn.fit(X, y, epochs=ep, verbose=1, validation_data=(X1, y1),
                                callbacks=[lr_scheduler, early_stopping])
        # Get mae and loss from history log
        mse = history.history['mse']
        loss = history.history['loss']
        # Get number of epochs
        # get the mse and loss values from the history object
        val_mse = history.history['val_mse']
        val_loss = history.history['val_loss']
        model_dnn.save('M_DDN_past8andNext24DNN.h5')
        # create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # plot the mse values on the first subplot
        ax1.plot(mse, label='Training MSE')
        ax1.plot(val_mse, label='Validation MSE')
        ax1.set_ylabel('MSE')
        ax1.legend()

        # plot the loss values on the second subplot
        ax2.plot(loss, label='Training Loss')
        ax2.plot(val_loss, label='Validation Loss')
        ax2.set_ylabel('Loss')
        ax2.legend()

        # set the x-axis label and tick marks
        plt.xlabel('Epoch')
        plt.xticks(range(len(mse)), range(1, len(mse) + 1))
        plt.show()
        end_time = time.time()
        times.append(end_time - start_time)

        # CNN
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
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
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
        model_cnn.save('M_CNN_past8andNext24DNN.h5')
        # create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # plot the mse values on the first subplot
        ax1.plot(mse, label='Training MSE')
        ax1.plot(val_mse, label='Validation MSE')
        ax1.set_ylabel('MSE')
        ax1.legend()

        # plot the loss values on the second subplot
        ax2.plot(loss, label='Training Loss')
        ax2.plot(val_loss, label='Validation Loss')
        ax2.set_ylabel('Loss')
        ax2.legend()

        # set the x-axis label and tick marks
        plt.xlabel('Epoch')
        plt.xticks(range(len(mse)), range(1, len(mse) + 1))
        plt.show()
        end_time = time.time()
        times.append(end_time - start_time)

        # RNN
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
        model_rnn.save('M_rnn_past8andNext24DNN.h5')
        # create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # plot the mse values on the first subplot
        ax1.plot(mse, label='Training MSE')
        ax1.plot(val_mse, label='Validation MSE')
        ax1.set_ylabel('MSE')
        ax1.legend()

        # plot the loss values on the second subplot
        ax2.plot(loss, label='Training Loss')
        ax2.plot(val_loss, label='Validation Loss')
        ax2.set_ylabel('Loss')
        ax2.legend()

        # set the x-axis label and tick marks
        plt.xlabel('Epoch')
        plt.xticks(range(len(mse)), range(1, len(mse) + 1))
        plt.show()
        end_time = time.time()
        times.append(end_time - start_time)

        # CNN_lstm
        # clear previous session
        # tf.keras.backend.clear_session()
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
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

        # print a summary of the model
        model_cnn_lstm.summary()

        # fit the model
        history = model_cnn_lstm.fit(X, y, epochs=ep, verbose=1, validation_data=(X1, y1),
                                     callbacks=[lr_scheduler, early_stopping])

        # save the model
        model_cnn_lstm.save('M_CNNLSTM_past8andNext24DNN.h5')

        # get the mse and loss values from the history object
        mse = history.history['mse']
        loss = history.history['loss']
        val_mse = history.history['val_mse']
        val_loss = history.history['val_loss']
        # create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # plot the mse values on the first subplot
        ax1.plot(mse, label='Training MSE')
        ax1.plot(val_mse, label='Validation MSE')
        ax1.set_ylabel('MSE')
        ax1.legend()

        # plot the loss values on the second subplot
        ax2.plot(loss, label='Training Loss')
        ax2.plot(val_loss, label='Validation Loss')
        ax2.set_ylabel('Loss')
        ax2.legend()

        # set the x-axis label and tick marks
        plt.xlabel('Epoch')
        plt.xticks(range(len(mse)), range(1, len(mse) + 1))
        plt.show()
        end_time = time.time()
        times.append(end_time - start_time)

        # RNN LSTM
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
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

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
        # create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # plot the mse values on the first subplot
        ax1.plot(mse, label='Training MSE')
        ax1.plot(val_mse, label='Validation MSE')
        ax1.set_ylabel('MSE')
        ax1.legend()

        # plot the loss values on the second subplot
        ax2.plot(loss, label='Training Loss')
        ax2.plot(val_loss, label='Validation Loss')
        ax2.set_ylabel('Loss')
        ax2.legend()

        # set the x-axis label and tick marks
        plt.xlabel('Epoch')
        plt.xticks(range(len(mse)), range(1, len(mse) + 1))
        plt.show()
        # Save the model
        hybrid_rnn_lstm_model.save('M_RNNLSTM_past8andNext24DNN.h5')
        end_time = time.time()
        times.append(end_time - start_time)

        # CNN RNN LSTM
        model_cnn_lstm_rnn = Sequential()

        # add a convolutional layer with 64 filters, a kernel size of 2, and ReLU activation
        model_cnn_lstm_rnn.add(
            Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps_in, n_features)))
        # add a max pooling layer with a pool size of 2
        model_cnn_lstm_rnn.add(MaxPooling1D(pool_size=2))
        # add a dropout layer with a rate of 0.3
        model_cnn_lstm_rnn.add(Dropout(0.3))

        # add an RNN layer with 128 units and return sequences
        model_cnn_lstm_rnn.add(SimpleRNN(128, activation='relu', return_sequences=True))
        # add an LSTM layer with 256 units and return sequences
        model_cnn_lstm_rnn.add(LSTM(256, activation='relu', return_sequences=True))
        # add another dropout layer with a rate of 0.3
        model_cnn_lstm_rnn.add(Dropout(0.3))

        # add a flatten layer to convert the output of the previous layer to a 1D tensor
        model_cnn_lstm_rnn.add(Flatten())
        # add a dense layer with 50 units and ReLU activation
        model_cnn_lstm_rnn.add(Dense(50, activation='relu'))
        # add a dense layer with n_steps_out units and ReLU activation
        model_cnn_lstm_rnn.add(Dense(n_steps_out, activation='relu'))

        # define the loss function as Huber
        loss = tf.keras.losses.Huber()
        # compile the model with adam optimizer and mse as a metric
        model_cnn_lstm_rnn.compile(loss=loss, optimizer='adam', metrics=['mse'])

        # define the learning rate scheduler and early stopping callbacks
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

        # print a summary of the model
        model_cnn_lstm_rnn.summary()

        # fit the model
        history = model_cnn_lstm_rnn.fit(X, y, epochs=ep, verbose=1, validation_data=(X1, y1),
                                         callbacks=[lr_scheduler, early_stopping])

        # save the model
        model_cnn_lstm_rnn.save('M_CNNRNNLSTM_past8andNext24DNN.h5')

        # get the mse and loss values from the history object
        mse = history.history['mse']
        loss = history.history['loss']
        val_mse = history.history['val_mse']
        val_loss = history.history['val_loss']
        # create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # plot the mse values on the first subplot
        ax1.plot(mse, label='Training MSE')
        ax1.plot(val_mse, label='Validation MSE')
        ax1.set_ylabel('MSE')
        ax1.legend()

        # plot the loss values on the second subplot
        ax2.plot(loss, label='Training Loss')
        ax2.plot(val_loss, label='Validation Loss')
        ax2.set_ylabel('Loss')
        ax2.legend()

        # set the x-axis label and tick marks
        plt.xlabel('Epoch')
        plt.xticks(range(len(mse)), range(1, len(mse) + 1))
        plt.show()
        end_time = time.time()
        times.append(end_time - start_time)

'''
'''
# Multi models 
imput_col = ['DryBulbTemperature','DirectNormalRadiation','GlobalHorizontalRadiation','SnowDepth','OccupancySchedule',
                         'WindDirection','Heating']
Heating_wrkd = load_data(imput_col
               ,
                        path=".\working_days.csv", verbose = True)

Heating_aftnon = load_data(
                         imput_col,
                        path=".\wafternoon.csv", verbose = False)

Heating_hol_wkn = load_data(
                         imput_col,
                        path=".\df_filtered_weekends_holidays.csv", verbose = False)

Heating_midday = load_data(
                         imput_col,
                        path=".\middy.csv", verbose = False)
Heating_morning = load_data(
                         imput_col,
                        path=".\morning_peak.csv", verbose = False)
Heating_night = load_data(
                         imput_col,
                        path=".\wnight_unoccupied_hours.csv", verbose = False)

List_S_ = [Heating_wrkd, Heating_aftnon, Heating_hol_wkn,Heating_midday,Heating_morning,Heating_night ]
###
# working days = All days exept holidays and weekendes 7 past 24 future
# night_unoccupied_hours = 6pm to 5 am , 6 past, 3 future
# miidy = 10-14
# afternoon: 15-17
# morning peak = 6-9
# holiday 7 past 10 future
class Metric:
    def __init__(self):
        self.metrics = pd.DataFrame(columns=['R-squared', 'MAE', 'MSE'])
    def add(self, y_true, y_pred):
        residuals = y_true - y_pred
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        mae = np.mean(np.abs(residuals))
        mse = np.mean(residuals ** 2)
        self.metrics.loc[len(self.metrics)] = [r_squared, mae, mse]
    def get(self):
        return self.metrics

class MetricsPlotter:
    def __init__(self, mse, val_mse, loss, val_loss):
        self.mse = mse
        self.val_mse = val_mse
        self.loss = loss
        self.val_loss = val_loss
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))

    def plot(self, title=None):
        # plot the mse values on the first subplot
        self.ax1.plot(self.mse, label='Training MSE')
        self.ax1.plot(self.val_mse, label='Validation MSE')
        self.ax1.set_ylabel('MSE')
        self.ax1.legend()
        # plot the loss values on the second subplot
        self.ax2.plot(self.loss, label='Training Loss')
        self.ax2.plot(self.val_loss, label='Validation Loss')
        self.ax2.set_ylabel('Loss')
        self.ax2.legend()

        # set the x-axis label and tick marks
        plt.xlabel('Epoch')
        plt.xticks(range(len(self.mse)), range(1, len(self.mse) + 1))

        # set the title of the plot
        if title is not None:
            plt.suptitle(title)

        plt.show()

R2_for_first_hour = []
times = []
# make the results reproducible

# starting
for i in range(6):
    start_time = time.time()
    s = List_S_[i]
    train_multi, test_multi = split_data2(s, 0.1,0.2)
    ep = 2
    metric = Metric()
    if i == 0:
        for j in s:
            vars()[f"in_seq0{j}"] = train_multi[j].values
            vars()[f"in_seq0{j}"].reshape((len(train_multi[j]), 1))
            vars()[f"in_seq0_test{j}"] = test_multi[j].values
            vars()[f"in_seq0_test{j}"].reshape((len(vars()[f"in_seq0_test{j}"]), 1))
        out_seq0 = in_seq0Heating
        dataset = hstack((in_seq0DirectNormalRadiation.reshape(-1, 1), in_seq0DryBulbTemperature.reshape(-1, 1),
                          in_seq0OccupancySchedule.reshape(-1, 1), in_seq0SnowDepth.reshape(-1, 1),
                          in_seq0GlobalHorizontalRadiation.reshape(-1, 1)
                          , in_seq0WindDirection.reshape(-1, 1), in_seq0Heating.reshape(-1, 1),
                          out_seq0.reshape(-1, 1)))
        out_seq0_test = in_seq0_testHeating
        dataset_test = hstack((in_seq0_testDirectNormalRadiation.reshape(-1, 1),
                               in_seq0_testDryBulbTemperature.reshape(-1, 1),
                               in_seq0_testOccupancySchedule.reshape(-1, 1), in_seq0_testSnowDepth.reshape(-1, 1),
                               in_seq0_testGlobalHorizontalRadiation.reshape(-1, 1)
                               , in_seq0_testWindDirection.reshape(-1, 1), in_seq0_testHeating.reshape(-1, 1),
                               out_seq0_test.reshape(-1, 1)))
        n_steps_in, n_steps_out = 7, 24
        X, y = split_sequences(dataset, n_steps_in, n_steps_out)
        X1, y1 = split_sequences(dataset_test, n_steps_in, n_steps_out)
        n_features = X.shape[2]
# LSTM
        np.random.seed(0)
        tf.random.set_seed(0)
        model = Sequential()
        model.add(LSTM(128, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
        model.add(Dropout(0.3))
        model.add(LSTM(256, activation='relu', return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(256, activation='relu', return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(128, activation='relu'))
        model.add(Dense(n_steps_out, activation='relu'))
        # set up the learning rate scheduler and early stopping
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
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
        model.save('0lstm_past8andNext24.h5')
#PostProcessing
        # create an instance of the MetricsPlotter class
        metrics_plotter = MetricsPlotter(mse, val_mse, loss, val_loss)
        metrics_plotter.plot(title='0lstm_past8andNext24')
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


#DNN
        #tf.keras.backend.clear_session()
        model_dnn = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(n_steps_in, n_features)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(n_steps_out, activation='relu')
        ], name='dnn')
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
        loss = tf.keras.losses.Huber()
        model_dnn.compile(loss=loss, optimizer='adam', metrics=['mse'])
        model_dnn.summary()
        # fit the model
        history = model_dnn.fit(X, y, epochs=ep, verbose=1, validation_data=(X1, y1),
                                callbacks=[lr_scheduler, early_stopping])
        # Get mae and loss from history log
        mse = history.history['mse']
        loss = history.history['loss']
        # Get number of epochs
        # get the mse and loss values from the history object
        val_mse = history.history['val_mse']
        val_loss = history.history['val_loss']
        model_dnn.save('0DDN_past8andNext24DNN.h5')
# PostProcessing
        # create an instance of the MetricsPlotter class
        metrics_plotter = MetricsPlotter(mse, val_mse, loss, val_loss)
        metrics_plotter.plot(title='0DDN_past8andNext24DNN')
        yhat1 = model_dnn.predict(X1, verbose=0)
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
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
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
        model_cnn.save('0CNN_past8andNext24DNN.h5')
# PostProcessing
        # create an instance of the MetricsPlotter class
        metrics_plotter = MetricsPlotter(mse, val_mse, loss, val_loss)
        metrics_plotter.plot(title='0CNN_past8andNext24DNN')
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
        model_rnn.save('0rnn_past8andNext24DNN.h5')
    # PostProcessing
        # create an instance of the MetricsPlotter class
        metrics_plotter = MetricsPlotter(mse, val_mse, loss, val_loss)
        metrics_plotter.plot(title='0rnn_past8andNext24DNN')

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


# CNN_lstm
        # clear previous session
        # tf.keras.backend.clear_session()
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
        #loss = tf.keras.losses.Huber()
        # compile the model with adam optimizer and mse as a metric
        model_cnn_lstm.compile(loss='MSE', optimizer='adam', metrics=['mse'])

        # define the learning rate scheduler and early stopping callbacks
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

        # print a summary of the model
        model_cnn_lstm.summary()

        # fit the model
        history = model_cnn_lstm.fit(X, y, epochs=ep, verbose=1, validation_data=(X1, y1),
                                     callbacks=[lr_scheduler, early_stopping])

        # save the model
        model_cnn_lstm.save('0CNNLSTM_past8andNext24DNN.h5')
    # PostProcessing
        # create an instance of the MetricsPlotter class
        metrics_plotter = MetricsPlotter(mse, val_mse, loss, val_loss)
        metrics_plotter.plot(title='0CNNLSTM_past8andNext24DNN')
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
        #loss = tf.keras.losses.Huber()
        hybrid_rnn_lstm_model.compile(loss='MSE', optimizer='adam', metrics=['mse'])

        # Define callbacks
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

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
        # create a figure with two subplots

        # Save the model
        hybrid_rnn_lstm_model.save('0RNNLSTM_past8andNext24DNN.h5')
    # PostProcessing
        # create an instance of the MetricsPlotter class
        metrics_plotter = MetricsPlotter(mse, val_mse, loss, val_loss)
        metrics_plotter.plot(title='0RNNLSTM_past8andNext24DNN')
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

        # CNN RNN LSTM
        model_cnn_lstm_rnn = Sequential()
        # add a convolutional layer with 64 filters, a kernel size of 2, and ReLU activation
        model_cnn_lstm_rnn.add(
            Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps_in, n_features)))
        # add a max pooling layer with a pool size of 2
        model_cnn_lstm_rnn.add(MaxPooling1D(pool_size=2))
        # add a dropout layer with a rate of 0.3
        model_cnn_lstm_rnn.add(Dropout(0.3))

        # add an RNN layer with 128 units and return sequences
        model_cnn_lstm_rnn.add(SimpleRNN(128, activation='relu', return_sequences=True))
        # add an LSTM layer with 256 units and return sequences
        model_cnn_lstm_rnn.add(LSTM(256, activation='relu', return_sequences=True))
        # add another dropout layer with a rate of 0.3
        model_cnn_lstm_rnn.add(Dropout(0.3))

        # add a flatten layer to convert the output of the previous layer to a 1D tensor
        model_cnn_lstm_rnn.add(Flatten())
        # add a dense layer with 50 units and ReLU activation
        model_cnn_lstm_rnn.add(Dense(50, activation='relu'))
        # add a dense layer with n_steps_out units and ReLU activation
        model_cnn_lstm_rnn.add(Dense(n_steps_out, activation='relu'))

        # define the loss function as Huber
        #loss = tf.keras.losses.Huber()
        # compile the model with adam optimizer and mse as a metric
        model_cnn_lstm_rnn.compile(loss='MSE', optimizer='adam', metrics=['mse'])

        # define the learning rate scheduler and early stopping callbacks
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

        # print a summary of the model
        model_cnn_lstm_rnn.summary()

        # fit the model
        history = model_cnn_lstm_rnn.fit(X, y, epochs=ep, verbose=1, validation_data=(X1, y1),
                                         callbacks=[lr_scheduler, early_stopping])

        # save the model
        model_cnn_lstm_rnn.save('0CNNRNNLSTM_past8andNext24DNN.h5')

        # get the mse and loss values from the history object
        mse = history.history['mse']
        loss = history.history['loss']
        val_mse = history.history['val_mse']
        val_loss = history.history['val_loss']
        # create a figure with two subplots
    # PostProcessing
        # create an instance of the MetricsPlotter class
        metrics_plotter = MetricsPlotter(mse, val_mse, loss, val_loss)
        metrics_plotter.plot(title='0CNNRNNLSTM_past8andNext24DNN')
        yhat1 = model_cnn_lstm_rnn.predict(X1, verbose=0)
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

    #afternoon: 15 - 17
    elif i == 1:
        for j in s:
            vars()[f"in_seq1{j}"] = train_multi[j].values
            vars()[f"in_seq1{j}"].reshape((len(train_multi[j]), 1))
            vars()[f"in_seq1_test{j}"] = test_multi[j].values
            vars()[f"in_seq1_test{j}"].reshape((len(vars()[f"in_seq1_test{j}"]), 1))

        out_seq1 = in_seq1Heating
        dataset = hstack((in_seq1DirectNormalRadiation.reshape(-1, 1), in_seq1DryBulbTemperature.reshape(-1, 1),
                          in_seq1OccupancySchedule.reshape(-1, 1), in_seq1SnowDepth.reshape(-1, 1),
                          in_seq1GlobalHorizontalRadiation.reshape(-1, 1)
                          , in_seq1WindDirection.reshape(-1, 1), in_seq1Heating.reshape(-1, 1),
                          out_seq1.reshape(-1, 1)))
        out_seq1_test = in_seq1_testHeating
        dataset_test = hstack((in_seq1_testDirectNormalRadiation.reshape(-1, 1),
                               in_seq1_testDryBulbTemperature.reshape(-1, 1),
                               in_seq1_testOccupancySchedule.reshape(-1, 1), in_seq1_testSnowDepth.reshape(-1, 1),
                               in_seq1_testGlobalHorizontalRadiation.reshape(-1, 1)
                               , in_seq1_testWindDirection.reshape(-1, 1), in_seq1_testHeating.reshape(-1, 1),
                               out_seq1_test.reshape(-1, 1)))
        n_steps_in, n_steps_out = 6, 3
        X, y = split_sequences(dataset, n_steps_in, n_steps_out)
        X1, y1 = split_sequences(dataset_test, n_steps_in, n_steps_out)
        n_features = X.shape[2]

        # LSTM
        model = Sequential()
        model.add(LSTM(128, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
        model.add(Dropout(0.3))
        model.add(LSTM(256, activation='relu', return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(256, activation='relu', return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(128, activation='relu'))
        model.add(Dense(n_steps_out, activation='relu'))
        # set up the learning rate scheduler and early stopping
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
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
        model.save('1lstm_past8andNext24.h5')
        # create a figure with two subplots
    # PostProcessing
        # create an instance of the MetricsPlotter class
        metrics_plotter = MetricsPlotter(mse, val_mse, loss, val_loss)
        metrics_plotter.plot(title='1lstm_past8andNext24')
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

# DNN
        tf.keras.backend.clear_session()
        model_dnn = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(n_steps_in, n_features)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(n_steps_out, activation='relu')
        ], name='dnn')
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
        loss = tf.keras.losses.Huber()
        model_dnn.compile(loss=loss, optimizer='adam', metrics=['mse'])
        model_dnn.summary()
        # fit the model
        history = model_dnn.fit(X, y, epochs=ep, verbose=1, validation_data=(X1, y1),
                                callbacks=[lr_scheduler, early_stopping])
        # Get mae and loss from history log
        mse = history.history['mse']
        loss = history.history['loss']
        # Get number of epochs
        # get the mse and loss values from the history object
        val_mse = history.history['val_mse']
        val_loss = history.history['val_loss']
        model_dnn.save('1DDN_past8andNext24DNN.h5')
        # create a figure with two subplots
    # PostProcessing
        # create an instance of the MetricsPlotter class
        metrics_plotter = MetricsPlotter(mse, val_mse, loss, val_loss)
        metrics_plotter.plot(title='1DDN_past8andNext24DNN')
        yhat1 = model_dnn.predict(X1, verbose=0)
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
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
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
        model_cnn.save('1CNN_past8andNext24DNN.h5')
    # PostProcessing
        # create an instance of the MetricsPlotter class
        metrics_plotter = MetricsPlotter(mse, val_mse, loss, val_loss)
        metrics_plotter.plot(title='1CNN_past8andNext24DNN')
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
        model_rnn.save('1rnn_past8andNext24DNN.h5')
# PostProcessing
        # create an instance of the MetricsPlotter class
        metrics_plotter = MetricsPlotter(mse, val_mse, loss, val_loss)
        metrics_plotter.plot(title='1rnn_past8andNext24DNN')
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

 # CNN_lstm
        # clear previous session
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
        #loss = tf.keras.losses.Huber()
        # compile the model with adam optimizer and mse as a metric
        model_cnn_lstm.compile(loss="MSE", optimizer='adam', metrics=['mse'])

        # define the learning rate scheduler and early stopping callbacks
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

        # print a summary of the model
        model_cnn_lstm.summary()

        # fit the model
        history = model_cnn_lstm.fit(X, y, epochs=ep, verbose=1, validation_data=(X1, y1),
                                     callbacks=[lr_scheduler, early_stopping])

        # save the model
        model_cnn_lstm.save('1CNNLSTM_past8andNext24DNN.h5')

        # get the mse and loss values from the history object
        mse = history.history['mse']
        loss = history.history['loss']
        val_mse = history.history['val_mse']
        val_loss = history.history['val_loss']
        # create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # plot the mse values on the first subplot
# PostProcessing
        # create an instance of the MetricsPlotter class
        metrics_plotter = MetricsPlotter(mse, val_mse, loss, val_loss)
        metrics_plotter.plot(title='1CNNLSTM_past8andNext24DNN')
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
        #loss = tf.keras.losses.Huber()
        hybrid_rnn_lstm_model.compile(loss='MSE', optimizer='adam', metrics=['mse'])

        # Define callbacks
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

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
        hybrid_rnn_lstm_model.save('1RNNLSTM_past8andNext24DNN.h5')
    # PostProcessing
        # create an instance of the MetricsPlotter class
        metrics_plotter = MetricsPlotter(mse, val_mse, loss, val_loss)
        metrics_plotter.plot(title='1RNNLSTM_past8andNext24DNN')
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


# CNN RNN LSTM
        model_cnn_lstm_rnn = Sequential()
        # add a convolutional layer with 64 filters, a kernel size of 2, and ReLU activation
        model_cnn_lstm_rnn.add(
            Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps_in, n_features)))
        # add a max pooling layer with a pool size of 2
        model_cnn_lstm_rnn.add(MaxPooling1D(pool_size=2))
        # add a dropout layer with a rate of 0.3
        model_cnn_lstm_rnn.add(Dropout(0.3))

        # add an RNN layer with 128 units and return sequences
        model_cnn_lstm_rnn.add(SimpleRNN(128, activation='relu', return_sequences=True))
        # add an LSTM layer with 256 units and return sequences
        model_cnn_lstm_rnn.add(LSTM(256, activation='relu', return_sequences=True))
        # add another dropout layer with a rate of 0.3
        model_cnn_lstm_rnn.add(Dropout(0.3))

        # add a flatten layer to convert the output of the previous layer to a 1D tensor
        model_cnn_lstm_rnn.add(Flatten())
        # add a dense layer with 50 units and ReLU activation
        model_cnn_lstm_rnn.add(Dense(50, activation='relu'))
        # add a dense layer with n_steps_out units and ReLU activation
        model_cnn_lstm_rnn.add(Dense(n_steps_out, activation='relu'))

        # define the loss function as Huber
        #loss = tf.keras.losses.Huber()
        # compile the model with adam optimizer and mse as a metric
        model_cnn_lstm_rnn.compile(loss='MSE', optimizer='adam', metrics=['mse'])

        # define the learning rate scheduler and early stopping callbacks
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

        # print a summary of the model
        model_cnn_lstm_rnn.summary()

        # fit the model
        history = model_cnn_lstm_rnn.fit(X, y, epochs=ep, verbose=1, validation_data=(X1, y1),
                                         callbacks=[lr_scheduler, early_stopping])

        # save the model
        model_cnn_lstm_rnn.save('1CNNRNNLSTM_past8andNext24DNN.h5')

    # PostProcessing
        # create an instance of the MetricsPlotter class
        metrics_plotter = MetricsPlotter(mse, val_mse, loss, val_loss)
        metrics_plotter.plot(title='1CNNRNNLSTM_past8andNext24DNN')
        yhat1 = model_cnn_lstm_rnn.predict(X1, verbose=0)
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

    if i == 2:
        for j in s:
            vars()[f"in_seq2{j}"] = train_multi[j].values
            vars()[f"in_seq2{j}"].reshape((len(train_multi[j]), 1))
            vars()[f"in_seq2_test{j}"] = test_multi[j].values
            vars()[f"in_seq2_test{j}"].reshape((len(vars()[f"in_seq2_test{j}"]), 1))

        out_seq2 = in_seq2Heating
        dataset = hstack((in_seq2DirectNormalRadiation.reshape(-1, 1), in_seq2DryBulbTemperature.reshape(-1, 1),
                          in_seq2OccupancySchedule.reshape(-1, 1), in_seq2SnowDepth.reshape(-1, 1),
                          in_seq2GlobalHorizontalRadiation.reshape(-1, 1)
                          , in_seq2WindDirection.reshape(-1, 1), in_seq2Heating.reshape(-1, 1),
                          out_seq2.reshape(-1, 1)))
        out_seq2_test = in_seq2_testHeating
        dataset_test = hstack((in_seq2_testDirectNormalRadiation.reshape(-1, 1),
                               in_seq2_testDryBulbTemperature.reshape(-1, 1),
                               in_seq2_testOccupancySchedule.reshape(-1, 1), in_seq2_testSnowDepth.reshape(-1, 1),
                               in_seq2_testGlobalHorizontalRadiation.reshape(-1, 1)
                               , in_seq2_testWindDirection.reshape(-1, 1), in_seq2_testHeating.reshape(-1, 1),
                               out_seq2_test.reshape(-1, 1)))
        n_steps_in, n_steps_out = 7, 10
        X, y = split_sequences(dataset, n_steps_in, n_steps_out)
        X1, y1 = split_sequences(dataset_test, n_steps_in, n_steps_out)
        n_features = X.shape[2]

        # LSTM
        model = Sequential()
        model.add(LSTM(128, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
        model.add(Dropout(0.3))
        model.add(LSTM(256, activation='relu', return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(256, activation='relu', return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(128, activation='relu'))
        model.add(Dense(n_steps_out, activation='relu'))
        # set up the learning rate scheduler and early stopping
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
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
        model.save('2lstm_past8andNext24.h5')
        # create a figure with two subplots
    # PostProcessing
        # create an instance of the MetricsPlotter class
        metrics_plotter = MetricsPlotter(mse, val_mse, loss, val_loss)
        metrics_plotter.plot(title='2lstm_past8andNext24')
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
# DNN
        tf.keras.backend.clear_session()
        model_dnn = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(n_steps_in, n_features)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(n_steps_out, activation='relu')
        ], name='dnn')
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
        loss = tf.keras.losses.Huber()
        model_dnn.compile(loss=loss, optimizer='adam', metrics=['mse'])
        model_dnn.summary()
        # fit the model
        history = model_dnn.fit(X, y, epochs=ep, verbose=1, validation_data=(X1, y1),
                                callbacks=[lr_scheduler, early_stopping])
        # Get mae and loss from history log
        mse = history.history['mse']
        loss = history.history['loss']
        # Get number of epochs
        # get the mse and loss values from the history object
        val_mse = history.history['val_mse']
        val_loss = history.history['val_loss']
        model_dnn.save('2DDN_past8andNext24DNN.h5')
        # create a figure with two subplots
    # PostProcessing
        # create an instance of the MetricsPlotter class
        metrics_plotter = MetricsPlotter(mse, val_mse, loss, val_loss)
        metrics_plotter.plot(title='2DDN_past8andNext24DNN')
        yhat1 = model_dnn.predict(X1, verbose=0)
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
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
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
        model_cnn.save('2CNN_past8andNext24DNN.h5')
        # create a figure with two subplots
    # PostProcessing
        # create an instance of the MetricsPlotter class
        metrics_plotter = MetricsPlotter(mse, val_mse, loss, val_loss)
        metrics_plotter.plot(title='2CNN_past8andNext24DNN')
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
        model_rnn.save('2rnn_past8andNext24DNN.h5')
        # create a figure with two subplots
    # PostProcessing
        # create an instance of the MetricsPlotter class
        metrics_plotter = MetricsPlotter(mse, val_mse, loss, val_loss)
        metrics_plotter.plot(title='2rnn_past8andNext24DNN')
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

# CNN_lstm
        # clear previous session
        # tf.keras.backend.clear_session()
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
        #loss = tf.keras.losses.Huber()
        # compile the model with adam optimizer and mse as a metric
        model_cnn_lstm.compile(loss='MSE', optimizer='adam', metrics=['mse'])

        # define the learning rate scheduler and early stopping callbacks
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

        # print a summary of the model
        model_cnn_lstm.summary()

        # fit the model
        history = model_cnn_lstm.fit(X, y, epochs=ep, verbose=1, validation_data=(X1, y1),
                                     callbacks=[lr_scheduler, early_stopping])

        # save the model
        model_cnn_lstm.save('2CNNLSTM_past8andNext24DNN.h5')

        # get the mse and loss values from the history object
        mse = history.history['mse']
        loss = history.history['loss']
        val_mse = history.history['val_mse']
        val_loss = history.history['val_loss']

    # PostProcessing
        # create an instance of the MetricsPlotter class
        metrics_plotter = MetricsPlotter(mse, val_mse, loss, val_loss)
        metrics_plotter.plot(title='2CNNLSTM_past8andNext24DNN')
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
        #loss = tf.keras.losses.Huber()
        hybrid_rnn_lstm_model.compile(loss='MSE', optimizer='adam', metrics=['mse'])

        # Define callbacks
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

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
        # create a figure with two subplots
        # Save the model
        hybrid_rnn_lstm_model.save('2RNNLSTM_past8andNext24DNN.h5')

    # PostProcessing
        # create an instance of the MetricsPlotter class
        metrics_plotter = MetricsPlotter(mse, val_mse, loss, val_loss)
        metrics_plotter.plot(title='2RNNLSTM_past8andNext24DNN')
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

# CNN RNN LSTM
        model_cnn_lstm_rnn = Sequential()
        # add a convolutional layer with 64 filters, a kernel size of 2, and ReLU activation
        model_cnn_lstm_rnn.add(
            Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps_in, n_features)))
        # add a max pooling layer with a pool size of 2
        model_cnn_lstm_rnn.add(MaxPooling1D(pool_size=2))
        # add a dropout layer with a rate of 0.3
        model_cnn_lstm_rnn.add(Dropout(0.3))

        # add an RNN layer with 128 units and return sequences
        model_cnn_lstm_rnn.add(SimpleRNN(128, activation='relu', return_sequences=True))
        # add an LSTM layer with 256 units and return sequences
        model_cnn_lstm_rnn.add(LSTM(256, activation='relu', return_sequences=True))
        # add another dropout layer with a rate of 0.3
        model_cnn_lstm_rnn.add(Dropout(0.3))

        # add a flatten layer to convert the output of the previous layer to a 1D tensor
        model_cnn_lstm_rnn.add(Flatten())
        # add a dense layer with 50 units and ReLU activation
        model_cnn_lstm_rnn.add(Dense(50, activation='relu'))
        # add a dense layer with n_steps_out units and ReLU activation
        model_cnn_lstm_rnn.add(Dense(n_steps_out, activation='relu'))

        # define the loss function as Huber
        #loss = tf.keras.losses.Huber()
        # compile the model with adam optimizer and mse as a metric
        model_cnn_lstm_rnn.compile(loss='MSE', optimizer='adam', metrics=['mse'])

        # define the learning rate scheduler and early stopping callbacks
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

        # print a summary of the model
        model_cnn_lstm_rnn.summary()

        # fit the model
        history = model_cnn_lstm_rnn.fit(X, y, epochs=ep, verbose=1, validation_data=(X1, y1),
                                         callbacks=[lr_scheduler, early_stopping])

        # save the model
        model_cnn_lstm_rnn.save('2CNNRNNLSTM_past8andNext24DNN.h5')

        # get the mse and loss values from the history object
        mse = history.history['mse']
        loss = history.history['loss']
        val_mse = history.history['val_mse']
        val_loss = history.history['val_loss']
        # create a figure with two subplots
    # PostProcessing
        # create an instance of the MetricsPlotter class
        metrics_plotter = MetricsPlotter(mse, val_mse, loss, val_loss)
        metrics_plotter.plot(title='2CNNRNNLSTM_past8andNext24DNN')
        yhat1 = model_cnn_lstm_rnn.predict(X1, verbose=0)
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

    # miidy = 10-14

    elif i == 3:
        for j in s:
            vars()[f"in_seq3{j}"] = train_multi[j].values
            vars()[f"in_seq3{j}"].reshape((len(train_multi[j]), 1))
            vars()[f"in_seq3_test{j}"] = test_multi[j].values
            vars()[f"in_seq3_test{j}"].reshape((len(vars()[f"in_seq3_test{j}"]), 1))

        out_seq3 = in_seq3Heating
        dataset = hstack((in_seq3DirectNormalRadiation.reshape(-1, 1), in_seq3DryBulbTemperature.reshape(-1, 1),
                          in_seq3OccupancySchedule.reshape(-1, 1), in_seq3SnowDepth.reshape(-1, 1),
                          in_seq3GlobalHorizontalRadiation.reshape(-1, 1)
                          , in_seq3WindDirection.reshape(-1, 1), in_seq3Heating.reshape(-1, 1),
                          out_seq3.reshape(-1, 1)))
        out_seq3_test = in_seq3_testHeating
        dataset_test = hstack((in_seq3_testDirectNormalRadiation.reshape(-1, 1),
                               in_seq3_testDryBulbTemperature.reshape(-1, 1),
                               in_seq3_testOccupancySchedule.reshape(-1, 1), in_seq3_testSnowDepth.reshape(-1, 1),
                               in_seq3_testGlobalHorizontalRadiation.reshape(-1, 1)
                               , in_seq3_testWindDirection.reshape(-1, 1), in_seq3_testHeating.reshape(-1, 1),
                               out_seq3_test.reshape(-1, 1)))
        n_steps_in, n_steps_out = 10, 5
        X, y = split_sequences(dataset, n_steps_in, n_steps_out)
        X1, y1 = split_sequences(dataset_test, n_steps_in, n_steps_out)
        n_features = X.shape[2]

        # LSTM
        model = Sequential()
        model.add(LSTM(128, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
        model.add(Dropout(0.3))
        model.add(LSTM(256, activation='relu', return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(256, activation='relu', return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(128, activation='relu'))
        model.add(Dense(n_steps_out, activation='relu'))
        # set up the learning rate scheduler and early stopping
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
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
        model.save('3lstm_past8andNext24.h5')
        # create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # plot the mse values on the first subplot
        ax1.plot(mse, label='Training MSE')
        ax1.plot(val_mse, label='Validation MSE')
        ax1.set_ylabel('MSE')
        ax1.legend()

        # plot the loss values on the second subplot
        ax2.plot(loss, label='Training Loss')
        ax2.plot(val_loss, label='Validation Loss')
        ax2.set_ylabel('Loss')
        ax2.legend()

        # set the x-axis label and tick marks
        plt.xlabel('Epoch')
        plt.xticks(range(len(mse)), range(1, len(mse) + 1))
        plt.show()
        end_time = time.time()
        times.append(end_time - start_time)

        # DNN
        tf.keras.backend.clear_session()
        model_dnn = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(n_steps_in, n_features)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(n_steps_out, activation='relu')
        ], name='dnn')
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
        loss = tf.keras.losses.Huber()
        model_dnn.compile(loss=loss, optimizer='adam', metrics=['mse'])
        model_dnn.summary()
        # fit the model
        history = model_dnn.fit(X, y, epochs=ep, verbose=1, validation_data=(X1, y1),
                                callbacks=[lr_scheduler, early_stopping])
        # Get mae and loss from history log
        mse = history.history['mse']
        loss = history.history['loss']
        # Get number of epochs
        # get the mse and loss values from the history object
        val_mse = history.history['val_mse']
        val_loss = history.history['val_loss']
        model_dnn.save('3DDN_past8andNext24DNN.h5')
        # create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # plot the mse values on the first subplot
        ax1.plot(mse, label='Training MSE')
        ax1.plot(val_mse, label='Validation MSE')
        ax1.set_ylabel('MSE')
        ax1.legend()

        # plot the loss values on the second subplot
        ax2.plot(loss, label='Training Loss')
        ax2.plot(val_loss, label='Validation Loss')
        ax2.set_ylabel('Loss')
        ax2.legend()

        # set the x-axis label and tick marks
        plt.xlabel('Epoch')
        plt.xticks(range(len(mse)), range(1, len(mse) + 1))
        plt.show()
        end_time = time.time()
        times.append(end_time - start_time)

        # CNN
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
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
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
        model_cnn.save('3CNN_past8andNext24DNN.h5')
        # create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # plot the mse values on the first subplot
        ax1.plot(mse, label='Training MSE')
        ax1.plot(val_mse, label='Validation MSE')
        ax1.set_ylabel('MSE')
        ax1.legend()

        # plot the loss values on the second subplot
        ax2.plot(loss, label='Training Loss')
        ax2.plot(val_loss, label='Validation Loss')
        ax2.set_ylabel('Loss')
        ax2.legend()

        # set the x-axis label and tick marks
        plt.xlabel('Epoch')
        plt.xticks(range(len(mse)), range(1, len(mse) + 1))
        plt.show()
        end_time = time.time()
        times.append(end_time - start_time)

        # RNN
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
        model_rnn.save('3rnn_past8andNext24DNN.h5')
        # create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # plot the mse values on the first subplot
        ax1.plot(mse, label='Training MSE')
        ax1.plot(val_mse, label='Validation MSE')
        ax1.set_ylabel('MSE')
        ax1.legend()

        # plot the loss values on the second subplot
        ax2.plot(loss, label='Training Loss')
        ax2.plot(val_loss, label='Validation Loss')
        ax2.set_ylabel('Loss')
        ax2.legend()

        # set the x-axis label and tick marks
        plt.xlabel('Epoch')
        plt.xticks(range(len(mse)), range(1, len(mse) + 1))
        plt.show()
        end_time = time.time()
        times.append(end_time - start_time)

        # CNN_lstm
        # clear previous session
        # tf.keras.backend.clear_session()
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
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

        # print a summary of the model
        model_cnn_lstm.summary()

        # fit the model
        history = model_cnn_lstm.fit(X, y, epochs=ep, verbose=1, validation_data=(X1, y1),
                                     callbacks=[lr_scheduler, early_stopping])

        # save the model
        model_cnn_lstm.save('3CNNLSTM_past8andNext24DNN.h5')

        # get the mse and loss values from the history object
        mse = history.history['mse']
        loss = history.history['loss']
        val_mse = history.history['val_mse']
        val_loss = history.history['val_loss']
        # create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # plot the mse values on the first subplot
        ax1.plot(mse, label='Training MSE')
        ax1.plot(val_mse, label='Validation MSE')
        ax1.set_ylabel('MSE')
        ax1.legend()

        # plot the loss values on the second subplot
        ax2.plot(loss, label='Training Loss')
        ax2.plot(val_loss, label='Validation Loss')
        ax2.set_ylabel('Loss')
        ax2.legend()

        # set the x-axis label and tick marks
        plt.xlabel('Epoch')
        plt.xticks(range(len(mse)), range(1, len(mse) + 1))
        plt.show()
        end_time = time.time()
        times.append(end_time - start_time)

        # RNN LSTM
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
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

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
        # create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # plot the mse values on the first subplot
        ax1.plot(mse, label='Training MSE')
        ax1.plot(val_mse, label='Validation MSE')
        ax1.set_ylabel('MSE')
        ax1.legend()

        # plot the loss values on the second subplot
        ax2.plot(loss, label='Training Loss')
        ax2.plot(val_loss, label='Validation Loss')
        ax2.set_ylabel('Loss')
        ax2.legend()

        # set the x-axis label and tick marks
        plt.xlabel('Epoch')
        plt.xticks(range(len(mse)), range(1, len(mse) + 1))
        plt.show()
        # Save the model
        hybrid_rnn_lstm_model.save('3RNNLSTM_past8andNext24DNN.h5')
        end_time = time.time()
        times.append(end_time - start_time)

        # CNN RNN LSTM
        model_cnn_lstm_rnn = Sequential()

        # add a convolutional layer with 64 filters, a kernel size of 2, and ReLU activation
        model_cnn_lstm_rnn.add(
            Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps_in, n_features)))
        # add a max pooling layer with a pool size of 2
        model_cnn_lstm_rnn.add(MaxPooling1D(pool_size=2))
        # add a dropout layer with a rate of 0.3
        model_cnn_lstm_rnn.add(Dropout(0.3))

        # add an RNN layer with 128 units and return sequences
        model_cnn_lstm_rnn.add(SimpleRNN(128, activation='relu', return_sequences=True))
        # add an LSTM layer with 256 units and return sequences
        model_cnn_lstm_rnn.add(LSTM(256, activation='relu', return_sequences=True))
        # add another dropout layer with a rate of 0.3
        model_cnn_lstm_rnn.add(Dropout(0.3))

        # add a flatten layer to convert the output of the previous layer to a 1D tensor
        model_cnn_lstm_rnn.add(Flatten())
        # add a dense layer with 50 units and ReLU activation
        model_cnn_lstm_rnn.add(Dense(50, activation='relu'))
        # add a dense layer with n_steps_out units and ReLU activation
        model_cnn_lstm_rnn.add(Dense(n_steps_out, activation='relu'))

        # define the loss function as Huber
        loss = tf.keras.losses.Huber()
        # compile the model with adam optimizer and mse as a metric
        model_cnn_lstm_rnn.compile(loss=loss, optimizer='adam', metrics=['mse'])

        # define the learning rate scheduler and early stopping callbacks
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

        # print a summary of the model
        model_cnn_lstm_rnn.summary()

        # fit the model
        history = model_cnn_lstm_rnn.fit(X, y, epochs=ep, verbose=1, validation_data=(X1, y1),
                                         callbacks=[lr_scheduler, early_stopping])

        # save the model
        model_cnn_lstm_rnn.save('3CNNRNNLSTM_past8andNext24DNN.h5')

        # get the mse and loss values from the history object
        mse = history.history['mse']
        loss = history.history['loss']
        val_mse = history.history['val_mse']
        val_loss = history.history['val_loss']
        # create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # plot the mse values on the first subplot
        ax1.plot(mse, label='Training MSE')
        ax1.plot(val_mse, label='Validation MSE')
        ax1.set_ylabel('MSE')
        ax1.legend()

        # plot the loss values on the second subplot
        ax2.plot(loss, label='Training Loss')
        ax2.plot(val_loss, label='Validation Loss')
        ax2.set_ylabel('Loss')
        ax2.legend()

        # set the x-axis label and tick marks
        plt.xlabel('Epoch')
        plt.xticks(range(len(mse)), range(1, len(mse) + 1))
        plt.show()
        end_time = time.time()
        times.append(end_time - start_time)



    ### morning peak = 6-9
    elif i == 4:
        for j in s:
            vars()[f"in_seq4{j}"] = train_multi[j].values
            vars()[f"in_seq4{j}"].reshape((len(train_multi[j]), 1))
            vars()[f"in_seq4_test{j}"] = test_multi[j].values
            vars()[f"in_seq4_test{j}"].reshape((len(vars()[f"in_seq4_test{j}"]), 1))

        out_seq4 = in_seq4Heating
        dataset = hstack((in_seq4DirectNormalRadiation.reshape(-1, 1), in_seq4DryBulbTemperature.reshape(-1, 1),
                          in_seq4OccupancySchedule.reshape(-1, 1), in_seq4SnowDepth.reshape(-1, 1),
                          in_seq4GlobalHorizontalRadiation.reshape(-1, 1)
                          , in_seq4WindDirection.reshape(-1, 1), in_seq4Heating.reshape(-1, 1),
                          out_seq4.reshape(-1, 1)))
        out_seq4_test = in_seq4_testHeating
        dataset_test = hstack((in_seq4_testDirectNormalRadiation.reshape(-1, 1),
                               in_seq4_testDryBulbTemperature.reshape(-1, 1),
                               in_seq4_testOccupancySchedule.reshape(-1, 1), in_seq4_testSnowDepth.reshape(-1, 1),
                               in_seq4_testGlobalHorizontalRadiation.reshape(-1, 1)
                               , in_seq4_testWindDirection.reshape(-1, 1), in_seq4_testHeating.reshape(-1, 1),
                               out_seq4_test.reshape(-1, 1)))
        n_steps_in, n_steps_out = 8, 4
        X, y = split_sequences(dataset, n_steps_in, n_steps_out)
        X1, y1 = split_sequences(dataset_test, n_steps_in, n_steps_out)
        n_features = X.shape[2]



        # LSTM
        model = Sequential()
        model.add(LSTM(128, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
        model.add(Dropout(0.3))
        model.add(LSTM(256, activation='relu', return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(256, activation='relu', return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(128, activation='relu'))
        model.add(Dense(n_steps_out, activation='relu'))
        # set up the learning rate scheduler and early stopping
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
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
        model.save('4lstm_past8andNext24.h5')
        # create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # plot the mse values on the first subplot
        ax1.plot(mse, label='Training MSE')
        ax1.plot(val_mse, label='Validation MSE')
        ax1.set_ylabel('MSE')
        ax1.legend()

        # plot the loss values on the second subplot
        ax2.plot(loss, label='Training Loss')
        ax2.plot(val_loss, label='Validation Loss')
        ax2.set_ylabel('Loss')
        ax2.legend()

        # set the x-axis label and tick marks
        plt.xlabel('Epoch')
        plt.xticks(range(len(mse)), range(1, len(mse) + 1))
        plt.show()
        end_time = time.time()
        times.append(end_time - start_time)

        # DNN
        tf.keras.backend.clear_session()
        model_dnn = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(n_steps_in, n_features)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(n_steps_out, activation='relu')
        ], name='dnn')
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
        loss = tf.keras.losses.Huber()
        model_dnn.compile(loss=loss, optimizer='adam', metrics=['mse'])
        model_dnn.summary()
        # fit the model
        history = model_dnn.fit(X, y, epochs=ep, verbose=1, validation_data=(X1, y1),
                                callbacks=[lr_scheduler, early_stopping])
        # Get mae and loss from history log
        mse = history.history['mse']
        loss = history.history['loss']
        # Get number of epochs
        # get the mse and loss values from the history object
        val_mse = history.history['val_mse']
        val_loss = history.history['val_loss']
        model_dnn.save('4DDN_past8andNext24DNN.h5')
        # create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # plot the mse values on the first subplot
        ax1.plot(mse, label='Training MSE')
        ax1.plot(val_mse, label='Validation MSE')
        ax1.set_ylabel('MSE')
        ax1.legend()

        # plot the loss values on the second subplot
        ax2.plot(loss, label='Training Loss')
        ax2.plot(val_loss, label='Validation Loss')
        ax2.set_ylabel('Loss')
        ax2.legend()

        # set the x-axis label and tick marks
        plt.xlabel('Epoch')
        plt.xticks(range(len(mse)), range(1, len(mse) + 1))
        plt.show()
        end_time = time.time()
        times.append(end_time - start_time)

        # CNN
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
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
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
        model_cnn.save('4CNN_past8andNext24DNN.h5')
        # create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # plot the mse values on the first subplot
        ax1.plot(mse, label='Training MSE')
        ax1.plot(val_mse, label='Validation MSE')
        ax1.set_ylabel('MSE')
        ax1.legend()

        # plot the loss values on the second subplot
        ax2.plot(loss, label='Training Loss')
        ax2.plot(val_loss, label='Validation Loss')
        ax2.set_ylabel('Loss')
        ax2.legend()

        # set the x-axis label and tick marks
        plt.xlabel('Epoch')
        plt.xticks(range(len(mse)), range(1, len(mse) + 1))
        plt.show()
        end_time = time.time()
        times.append(end_time - start_time)

        # RNN
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
        model_rnn.save('4rnn_past8andNext24DNN.h5')
        # create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # plot the mse values on the first subplot
        ax1.plot(mse, label='Training MSE')
        ax1.plot(val_mse, label='Validation MSE')
        ax1.set_ylabel('MSE')
        ax1.legend()

        # plot the loss values on the second subplot
        ax2.plot(loss, label='Training Loss')
        ax2.plot(val_loss, label='Validation Loss')
        ax2.set_ylabel('Loss')
        ax2.legend()

        # set the x-axis label and tick marks
        plt.xlabel('Epoch')
        plt.xticks(range(len(mse)), range(1, len(mse) + 1))
        plt.show()
        end_time = time.time()
        times.append(end_time - start_time)

        # CNN_lstm
        # clear previous session
        # tf.keras.backend.clear_session()
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
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

        # print a summary of the model
        model_cnn_lstm.summary()

        # fit the model
        history = model_cnn_lstm.fit(X, y, epochs=ep, verbose=1, validation_data=(X1, y1),
                                     callbacks=[lr_scheduler, early_stopping])

        # save the model
        model_cnn_lstm.save('4CNNLSTM_past8andNext24DNN.h5')

        # get the mse and loss values from the history object
        mse = history.history['mse']
        loss = history.history['loss']
        val_mse = history.history['val_mse']
        val_loss = history.history['val_loss']
        # create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # plot the mse values on the first subplot
        ax1.plot(mse, label='Training MSE')
        ax1.plot(val_mse, label='Validation MSE')
        ax1.set_ylabel('MSE')
        ax1.legend()

        # plot the loss values on the second subplot
        ax2.plot(loss, label='Training Loss')
        ax2.plot(val_loss, label='Validation Loss')
        ax2.set_ylabel('Loss')
        ax2.legend()

        # set the x-axis label and tick marks
        plt.xlabel('Epoch')
        plt.xticks(range(len(mse)), range(1, len(mse) + 1))
        plt.show()
        end_time = time.time()
        times.append(end_time - start_time)

        # RNN LSTM
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
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

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
        # create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # plot the mse values on the first subplot
        ax1.plot(mse, label='Training MSE')
        ax1.plot(val_mse, label='Validation MSE')
        ax1.set_ylabel('MSE')
        ax1.legend()

        # plot the loss values on the second subplot
        ax2.plot(loss, label='Training Loss')
        ax2.plot(val_loss, label='Validation Loss')
        ax2.set_ylabel('Loss')
        ax2.legend()

        # set the x-axis label and tick marks
        plt.xlabel('Epoch')
        plt.xticks(range(len(mse)), range(1, len(mse) + 1))
        plt.show()
        # Save the model
        hybrid_rnn_lstm_model.save('4RNNLSTM_past8andNext24DNN.h5')
        end_time = time.time()
        times.append(end_time - start_time)

        # CNN RNN LSTM
        model_cnn_lstm_rnn = Sequential()

        # add a convolutional layer with 64 filters, a kernel size of 2, and ReLU activation
        model_cnn_lstm_rnn.add(
            Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps_in, n_features)))
        # add a max pooling layer with a pool size of 2
        model_cnn_lstm_rnn.add(MaxPooling1D(pool_size=2))
        # add a dropout layer with a rate of 0.3
        model_cnn_lstm_rnn.add(Dropout(0.3))

        # add an RNN layer with 128 units and return sequences
        model_cnn_lstm_rnn.add(SimpleRNN(128, activation='relu', return_sequences=True))
        # add an LSTM layer with 256 units and return sequences
        model_cnn_lstm_rnn.add(LSTM(256, activation='relu', return_sequences=True))
        # add another dropout layer with a rate of 0.3
        model_cnn_lstm_rnn.add(Dropout(0.3))

        # add a flatten layer to convert the output of the previous layer to a 1D tensor
        model_cnn_lstm_rnn.add(Flatten())
        # add a dense layer with 50 units and ReLU activation
        model_cnn_lstm_rnn.add(Dense(50, activation='relu'))
        # add a dense layer with n_steps_out units and ReLU activation
        model_cnn_lstm_rnn.add(Dense(n_steps_out, activation='relu'))

        # define the loss function as Huber
        loss = tf.keras.losses.Huber()
        # compile the model with adam optimizer and mse as a metric
        model_cnn_lstm_rnn.compile(loss=loss, optimizer='adam', metrics=['mse'])

        # define the learning rate scheduler and early stopping callbacks
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

        # print a summary of the model
        model_cnn_lstm_rnn.summary()

        # fit the model
        history = model_cnn_lstm_rnn.fit(X, y, epochs=ep, verbose=1, validation_data=(X1, y1),
                                         callbacks=[lr_scheduler, early_stopping])

        # save the model
        model_cnn_lstm_rnn.save('4CNNRNNLSTM_past8andNext24DNN.h5')

        # get the mse and loss values from the history object
        mse = history.history['mse']
        loss = history.history['loss']
        val_mse = history.history['val_mse']
        val_loss = history.history['val_loss']
        # create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # plot the mse values on the first subplot
        ax1.plot(mse, label='Training MSE')
        ax1.plot(val_mse, label='Validation MSE')
        ax1.set_ylabel('MSE')
        ax1.legend()

        # plot the loss values on the second subplot
        ax2.plot(loss, label='Training Loss')
        ax2.plot(val_loss, label='Validation Loss')
        ax2.set_ylabel('Loss')
        ax2.legend()

        # set the x-axis label and tick marks
        plt.xlabel('Epoch')
        plt.xticks(range(len(mse)), range(1, len(mse) + 1))
        plt.show()
        end_time = time.time()
        times.append(end_time - start_time)



# 6pm to 5 am
    elif i == 5:
        for j in s:
            vars()[f"in_seq5{j}"] = train_multi[j].values
            vars()[f"in_seq5{j}"].reshape((len(train_multi[j]), 1))
            vars()[f"in_seq5_test{j}"] = test_multi[j].values
            vars()[f"in_seq5_test{j}"].reshape((len(vars()[f"in_seq5_test{j}"]), 1))

        out_seq5 = in_seq5Heating
        dataset = hstack((in_seq5DirectNormalRadiation.reshape(-1, 1), in_seq5DryBulbTemperature.reshape(-1, 1),
                          in_seq5OccupancySchedule.reshape(-1, 1), in_seq5SnowDepth.reshape(-1, 1),
                          in_seq5GlobalHorizontalRadiation.reshape(-1, 1)
                          , in_seq5WindDirection.reshape(-1, 1), in_seq5Heating.reshape(-1, 1),
                          out_seq5.reshape(-1, 1)))
        out_seq5_test = in_seq5_testHeating
        dataset_test = hstack((in_seq5_testDirectNormalRadiation.reshape(-1, 1),
                               in_seq5_testDryBulbTemperature.reshape(-1, 1),
                               in_seq5_testOccupancySchedule.reshape(-1, 1), in_seq5_testSnowDepth.reshape(-1, 1),
                               in_seq5_testGlobalHorizontalRadiation.reshape(-1, 1)
                               , in_seq5_testWindDirection.reshape(-1, 1), in_seq5_testHeating.reshape(-1, 1),
                               out_seq5_test.reshape(-1, 1)))
        n_steps_in, n_steps_out = 11, 11
        X, y = split_sequences(dataset, n_steps_in, n_steps_out)
        X1, y1 = split_sequences(dataset_test, n_steps_in, n_steps_out)
        n_features = X.shape[2]

        # LSTM
        model = Sequential()
        model.add(LSTM(128, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
        model.add(Dropout(0.3))
        model.add(LSTM(256, activation='relu', return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(256, activation='relu', return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(128, activation='relu'))
        model.add(Dense(n_steps_out, activation='relu'))
        # set up the learning rate scheduler and early stopping
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
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
        model.save('5lstm_past8andNext24.h5')
        # create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # plot the mse values on the first subplot
        ax1.plot(mse, label='Training MSE')
        ax1.plot(val_mse, label='Validation MSE')
        ax1.set_ylabel('MSE')
        ax1.legend()

        # plot the loss values on the second subplot
        ax2.plot(loss, label='Training Loss')
        ax2.plot(val_loss, label='Validation Loss')
        ax2.set_ylabel('Loss')
        ax2.legend()

        # set the x-axis label and tick marks
        plt.xlabel('Epoch')
        plt.xticks(range(len(mse)), range(1, len(mse) + 1))
        plt.show()
        end_time = time.time()
        times.append(end_time - start_time)

        # DNN
        tf.keras.backend.clear_session()
        model_dnn = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(n_steps_in, n_features)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(n_steps_out, activation='relu')
        ], name='dnn')
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
        loss = tf.keras.losses.Huber()
        model_dnn.compile(loss=loss, optimizer='adam', metrics=['mse'])
        model_dnn.summary()
        # fit the model
        history = model_dnn.fit(X, y, epochs=ep, verbose=1, validation_data=(X1, y1),
                                callbacks=[lr_scheduler, early_stopping])
        # Get mae and loss from history log
        mse = history.history['mse']
        loss = history.history['loss']
        # Get number of epochs
        # get the mse and loss values from the history object
        val_mse = history.history['val_mse']
        val_loss = history.history['val_loss']
        model_dnn.save('5DDN_past8andNext24DNN.h5')
        # create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # plot the mse values on the first subplot
        ax1.plot(mse, label='Training MSE')
        ax1.plot(val_mse, label='Validation MSE')
        ax1.set_ylabel('MSE')
        ax1.legend()

        # plot the loss values on the second subplot
        ax2.plot(loss, label='Training Loss')
        ax2.plot(val_loss, label='Validation Loss')
        ax2.set_ylabel('Loss')
        ax2.legend()

        # set the x-axis label and tick marks
        plt.xlabel('Epoch')
        plt.xticks(range(len(mse)), range(1, len(mse) + 1))
        plt.show()
        end_time = time.time()
        times.append(end_time - start_time)

        # CNN
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
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
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
        model_cnn.save('5CNN_past8andNext24DNN.h5')
        # create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # plot the mse values on the first subplot
        ax1.plot(mse, label='Training MSE')
        ax1.plot(val_mse, label='Validation MSE')
        ax1.set_ylabel('MSE')
        ax1.legend()

        # plot the loss values on the second subplot
        ax2.plot(loss, label='Training Loss')
        ax2.plot(val_loss, label='Validation Loss')
        ax2.set_ylabel('Loss')
        ax2.legend()

        # set the x-axis label and tick marks
        plt.xlabel('Epoch')
        plt.xticks(range(len(mse)), range(1, len(mse) + 1))
        plt.show()
        end_time = time.time()
        times.append(end_time - start_time)

        # RNN
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
        model_rnn.save('5rnn_past8andNext24DNN.h5')
        # create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # plot the mse values on the first subplot
        ax1.plot(mse, label='Training MSE')
        ax1.plot(val_mse, label='Validation MSE')
        ax1.set_ylabel('MSE')
        ax1.legend()

        # plot the loss values on the second subplot
        ax2.plot(loss, label='Training Loss')
        ax2.plot(val_loss, label='Validation Loss')
        ax2.set_ylabel('Loss')
        ax2.legend()

        # set the x-axis label and tick marks
        plt.xlabel('Epoch')
        plt.xticks(range(len(mse)), range(1, len(mse) + 1))
        plt.show()
        end_time = time.time()
        times.append(end_time - start_time)

        # CNN_lstm
        # clear previous session
        # tf.keras.backend.clear_session()
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
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

        # print a summary of the model
        model_cnn_lstm.summary()

        # fit the model
        history = model_cnn_lstm.fit(X, y, epochs=ep, verbose=1, validation_data=(X1, y1),
                                     callbacks=[lr_scheduler, early_stopping])

        # save the model
        model_cnn_lstm.save('5CNNLSTM_past8andNext24DNN.h5')

        # get the mse and loss values from the history object
        mse = history.history['mse']
        loss = history.history['loss']
        val_mse = history.history['val_mse']
        val_loss = history.history['val_loss']
        # create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # plot the mse values on the first subplot
        ax1.plot(mse, label='Training MSE')
        ax1.plot(val_mse, label='Validation MSE')
        ax1.set_ylabel('MSE')
        ax1.legend()

        # plot the loss values on the second subplot
        ax2.plot(loss, label='Training Loss')
        ax2.plot(val_loss, label='Validation Loss')
        ax2.set_ylabel('Loss')
        ax2.legend()

        # set the x-axis label and tick marks
        plt.xlabel('Epoch')
        plt.xticks(range(len(mse)), range(1, len(mse) + 1))
        plt.show()
        end_time = time.time()
        times.append(end_time - start_time)

        # RNN LSTM
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
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

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
        # create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # plot the mse values on the first subplot
        ax1.plot(mse, label='Training MSE')
        ax1.plot(val_mse, label='Validation MSE')
        ax1.set_ylabel('MSE')
        ax1.legend()

        # plot the loss values on the second subplot
        ax2.plot(loss, label='Training Loss')
        ax2.plot(val_loss, label='Validation Loss')
        ax2.set_ylabel('Loss')
        ax2.legend()

        # set the x-axis label and tick marks
        plt.xlabel('Epoch')
        plt.xticks(range(len(mse)), range(1, len(mse) + 1))
        plt.show()
        # Save the model
        hybrid_rnn_lstm_model.save('5RNNLSTM_past8andNext24DNN.h5')
        end_time = time.time()
        times.append(end_time - start_time)

        # CNN RNN LSTM
        model_cnn_lstm_rnn = Sequential()

        # add a convolutional layer with 64 filters, a kernel size of 2, and ReLU activation
        model_cnn_lstm_rnn.add(
            Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps_in, n_features)))
        # add a max pooling layer with a pool size of 2
        model_cnn_lstm_rnn.add(MaxPooling1D(pool_size=2))
        # add a dropout layer with a rate of 0.3
        model_cnn_lstm_rnn.add(Dropout(0.3))

        # add an RNN layer with 128 units and return sequences
        model_cnn_lstm_rnn.add(SimpleRNN(128, activation='relu', return_sequences=True))
        # add an LSTM layer with 256 units and return sequences
        model_cnn_lstm_rnn.add(LSTM(256, activation='relu', return_sequences=True))
        # add another dropout layer with a rate of 0.3
        model_cnn_lstm_rnn.add(Dropout(0.3))

        # add a flatten layer to convert the output of the previous layer to a 1D tensor
        model_cnn_lstm_rnn.add(Flatten())
        # add a dense layer with 50 units and ReLU activation
        model_cnn_lstm_rnn.add(Dense(50, activation='relu'))
        # add a dense layer with n_steps_out units and ReLU activation
        model_cnn_lstm_rnn.add(Dense(n_steps_out, activation='relu'))

        # define the loss function as Huber
        loss = tf.keras.losses.Huber()
        # compile the model with adam optimizer and mse as a metric
        model_cnn_lstm_rnn.compile(loss=loss, optimizer='adam', metrics=['mse'])

        # define the learning rate scheduler and early stopping callbacks
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

        # print a summary of the model
        model_cnn_lstm_rnn.summary()

        # fit the model
        history = model_cnn_lstm_rnn.fit(X, y, epochs=ep, verbose=1, validation_data=(X1, y1),
                                         callbacks=[lr_scheduler, early_stopping])

        # save the model
        model_cnn_lstm_rnn.save('5CNNRNNLSTM_past8andNext24DNN.h5')

        # get the mse and loss values from the history object
        mse = history.history['mse']
        loss = history.history['loss']
        val_mse = history.history['val_mse']
        val_loss = history.history['val_loss']
        # create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # plot the mse values on the first subplot
        ax1.plot(mse, label='Training MSE')
        ax1.plot(val_mse, label='Validation MSE')
        ax1.set_ylabel('MSE')
        ax1.legend()

        # plot the loss values on the second subplot
        ax2.plot(loss, label='Training Loss')
        ax2.plot(val_loss, label='Validation Loss')
        ax2.set_ylabel('Loss')
        ax2.legend()

        # set the x-axis label and tick marks
        plt.xlabel('Epoch')
        plt.xticks(range(len(mse)), range(1, len(mse) + 1))
        plt.show()
        end_time = time.time()
        times.append(end_time - start_time)
'''
'''
      ## Lstm with attention layer
        model_at_ls = Sequential()
        model_at_ls.add(LSTM(128, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
        model_at_ls.add(Dropout(0.3))
        model_at_ls.add(LSTM(256, activation='relu', return_sequences=True))
        model_at_ls.add(Dropout(0.3))
        model_at_ls.add(LSTM(256, activation='relu', return_sequences=True))
        model_at_ls.add(Dropout(0.3))
        model_at_ls.add(LSTM(128, activation='relu', return_sequences=True))
        model_at_ls.add(AttentionLayer())  # Add Attention layer
        model_at_ls.add(Dense(n_steps_out, activation='relu'))
        # set up the learning rate scheduler and early stopping
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
        # compile the model with the Adam optimizer and Huber loss
        model_at_ls.compile(optimizer=Adam(lr=1e-4), loss=tf.keras.losses.Huber(), metrics=['mse'])
        # train the model with early stopping and the learning rate scheduler
        history = model_at_ls.fit(X, y, epochs=ep, verbose=1, validation_data=(X1, y1),
                                  callbacks=[lr_scheduler, early_stopping])
        # Get mae and loss from history log
        mse = history.history['mse']
        loss = history.history['loss']
        # Get number of epochs
        # get the mse and loss values from the history object
        val_mse = history.history['val_mse']
        val_loss = history.history['val_loss']
        # save model
        model_at_ls.save('timeasfeature_lstm_at_past8andNext24.h5')

        # PostProcessing
        # create an instance of the MetricsPlotter class
        metrics_plotter = MetricsPlotter(mse, val_mse, loss, val_loss)
        metrics_plotter.plot(title='timeasfeature_lstm_at_past8andNext24')
        yhat1 = model_at_ls.predict(X1, verbose=0)
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
'''


