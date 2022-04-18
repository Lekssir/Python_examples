import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from basic_dl import BasicMethodsDL as b
from nolitsa import utils
from PhaseSpaceRecovery import reconstruct
from tensorflow import keras
from tensorflow.keras import layers
from sklearn import preprocessing

data, ind_list = reconstruct(number=5)

window_size = 32
batch_size = 16
len_dim = 1

split_time = round(data.size() * 0.8)

#data = utils.rescale(data, interval=(-10, 10))
data = data[14:]


ind1 = ind_list[0][14:]
len_data = data.size

dataset = b.window_dataset_f(data, window_size=window_size, batch_size=batch_size)
#train_ds, valid_ds = b.split_dataset(dataset, 0.2)

data_normaliser = preprocessing.MinMaxScaler()
data = data.reshape(-1, 1)
data = data_normaliser.fit_transform(data)

print(data.tail())

ind_normalizer = preprocessing.MinMaxScaler()
#ind = ind.reshape(-1, 1)
ind_norm = ind_normalizer.fit_transform(ind)

features = np.array(
    [data[i: i + window_size].copy() for i in range(len_data - window_size)])
rsi_ind = np.array(
    [ind1[i: i + window_size].copy() for i in range(len_data - window_size)])

features2 = np.array(
    [ind_norm[i: i + window_size].copy() for i in range(len(ind_norm) - window_size)])

targets = np.array(
    [data[i + window_size].copy() for i in range(len_data - window_size)])

print(features.shape, rsi_ind.shape)

feature_list = [features, features2]  ##[features, features2] rsi_ind
#feature_list = features
#dt = preprocessing.MinMaxScaler()
#dt.fit(series[split_time:])
time = [x for x in range(split_time, len(data))]


keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

input1 = keras.Input(shape=(window_size, 1))

#x1 = layers.Conv1D(filters=32, kernel_size=4, strides=1, dilation_rate=1, padding="causal", activation="relu")(input1)
#x1 = layers.Conv1D(filters=32, kernel_size=4, strides=1, dilation_rate=2, padding="causal", activation="relu")(x1)
#x1 = layers.Conv1D(filters=32, kernel_size=4, strides=1, dilation_rate=4, padding="causal", activation="relu")(x1)
#x1 = layers.Conv1D(filters=32, kernel_size=4, strides=1, dilation_rate=8, padding="causal", activation="relu")(x1)
#x1 = layers.Conv1D(filters=32, kernel_size=4, strides=1, padding="causal", activation="relu")(x1)
#x1 = layers.MaxPooling1D(pool_size=2, strides=None, padding="valid", data_format="channels_last")(x1)
#x1 = layers.LSTM(64)(input1)
#x1 = layers.Dense(32, activation="relu")(input1)
#x1 = layers.Dense(16, activation="relu")(x1)
#x1 = layers.Dense(8, activation="relu")(x1)
#x1_out = layers.Dense(1, activation='tanh')(x1)

x = layers.LSTM(64, name='lstm_01')(input1)
x = layers.Dropout(0.2, name='lstm_dropout_01')(x)
x = layers.Dense(32, name='dense_01')(x)
x = layers.Activation('sigmoid', name='sigmoid_01')(x)
x = layers.Dense(16, name='dense_02')(x)
x = layers.Activation('sigmoid', name='sigmoid_02')(x)
x = keras.Model(inputs=input1, outputs=x)
#x = layers.Dense(1, name='dense_1')(x)
#x1_out = layers.Activation('linear', name='linear_output')(x)

input2 = keras.Input(shape=(window_size, 1))
y = layers.LSTM(32, name='lstm_11')(input2)
#y = layers.Dropout(0.2, name='lstm_dropout_0')(x)
y = layers.Dense(16, name='dense_11')(y)
y = layers.Activation('sigmoid', name='sigmoid_11')(y)
y = keras.Model(inputs=input2, outputs=y)

combined = layers.concatenate([x.output, y.output])
z = layers.Dense(32, name='Conc_dense')(combined)
z = layers.Activation('sigmoid', name='sigmoid_3')(z)
z = layers.Dense(16, name='Conc_dense2')(z)
z = layers.Activation('sigmoid', name='sigmoid_4')(z)
z = layers.Dense(2, name='Conc_dense3')(z)
z = layers.Activation('sigmoid', name='sigmoid_5')(z)
z = layers.Dense(1, name='Conc_dense4')(z)
z = layers.Activation('linear', name='linear_output')(z)

model = keras.Model(inputs=[x.input, y.input], outputs=z, name='0.2_test')
#model = b.create_model('rnn')
#keras.utils.plot_model(model, "multi_input_and_output_model.png", show_shapes=True)

model.summary()

#keras.utils.plot_model(model, 'model_info.png', show_shapes=True)
#b.lr_plot(model, dataset, None, lr0=1e-2, nof_epochs=100, tf_batching=True)
b.lr_plot(model, feature_list, targets, lr0=1e-5, nof_epochs=100)
model = b.train(model, feature_list, targets, lr=1e-3, tf_batching=False)
#model = keras.models.load_model("my_checkpoint")
forecast = b.forecast_t(model, feature_list, targets, batch_size)
b.forecast_full(model, dataset,None, time, window_size, None, batch_size, tf_batching=True)

#market_sim = TradeModel()
#print(time_valid)
#market_sim.simulate(data, forecast, time)
#market_sim.plot_history()
