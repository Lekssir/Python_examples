import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
keras = tf.keras


class BasicMethodsDL:

    @staticmethod
    def datal():
        data = pd.read_csv('btc_usd_daily.csv')
        return data['VWAP'].to_numpy()

    @staticmethod
    def data_load(symbol):
        data = pd.read_csv(f'./{symbol}.csv')
        data = data.drop('date', axis=1)
        #print(data)
        #data = data.drop('Adj Close', axis=1)
        #data = data.drop(0, axis=0)
        ind = data.pop('SMA')
        return data.to_numpy(), ind.to_numpy()


    @staticmethod
    def plot_series(time, series, format="-", start=0, end=None, label=None):
        plt.plot(time[start:end], series[start:end], format, label=label)
        plt.xlabel("Time")
        plt.ylabel("Value")
        if label:
            plt.legend(fontsize=14)
        plt.grid(True)
        #plt.show()

    @staticmethod
    def window_dataset(series, window_size, batch_size=32,
                       shuffle_buffer=1000):
        dataset = tf.data.Dataset.from_tensor_slices(series)
        dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
        dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
        dataset = dataset.shuffle(shuffle_buffer)
        feature = dataset.map(lambda window: window[:-1])
        target = dataset.map(lambda window: [window[-1, 0]])
        dataset = dataset.batch(batch_size).prefetch(1)
        return feature, target

    @staticmethod
    def window_dataset_f(series, window_size, batch_size=16,
                       shuffle_buffer=1000):
        dataset = tf.data.Dataset.from_tensor_slices(series)
        dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
        dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
        #print(dataset)
        dataset = dataset.shuffle(shuffle_buffer)
        dataset = dataset.map(lambda window: (window[:-1], [window[-1]]))
        print(dataset)
        #print(list(dataset.as_numpy_iterator()))
        dataset = dataset.batch(batch_size).prefetch(1)
        return dataset

    @staticmethod
    def split_dataset(dataset: tf.data.Dataset, validation_data_fraction: float):

        validation_data_percent = round(validation_data_fraction * 100)
        if not (0 <= validation_data_percent <= 100):
            raise ValueError("invalid data fraction")

        dataset = dataset.enumerate()
        train_dataset = dataset.filter(lambda f, data: f % 100 > validation_data_percent)
        validation_dataset = dataset.filter(lambda f, data: f % 100 <= validation_data_percent)
        train_dataset = train_dataset.map(lambda f, data: data)
        validation_dataset = validation_dataset.map(lambda f, data: data)

        return train_dataset, validation_dataset

    @staticmethod
    def model_forecast(model, valid_list, batch_size=1, multi_step=False):
        #ds = tf.data.Dataset.from_tensor_slices(series)
        #ds = ds.window(window_size, shift=1, drop_remainder=True)
        #ds = ds.flat_map(lambda w: w.batch(window_size))
        #ds = ds.batch(batch_size).prefetch(1)
        forecast = model.predict(x=valid_list, batch_size=batch_size)

        if multi_step:
            multi_list = []
            for i in range(len(valid_list) - 1):
                wind = valid_list[i+1]
                wind[-1] = forecast[i]
                multi_list = multi_list.append(wind)
            wind = valid_list[-1][1:-1]
            wind = wind.append(forecast[-1])
            multi_list = multi_list.append(wind)
            forecast_multi = model.predict(x=multi_list, batch_size=batch_size)
            return forecast, forecast_multi

        return forecast

    @staticmethod
    def lr_plot(model, feature_list, targets, lr0=1e-5, nof_epochs=10, tf_batching=False):
        lr_schedule = keras.callbacks.LearningRateScheduler(lambda epoch: lr0 * 10**(epoch / 30))
        optimizer = keras.optimizers.SGD(lr=lr0, momentum=0.9)
        #optimizer = keras.optimizers.Adam(lr=lr0)
        model.compile(loss=keras.losses.MeanSquaredError(),
                      optimizer=optimizer,
                      metrics=["accuracy"])
        if tf_batching:
            history = model.fit(feature_list, batch_size=16, epochs=nof_epochs,
                                verbose=2, callbacks=[lr_schedule])
        else:
            history = model.fit(x=feature_list, y=targets, batch_size=16, validation_split=0.2, epochs=nof_epochs,
                                verbose=2, callbacks=[lr_schedule])
        plt.figure(figsize=(10, 6))
        #print(history.history["loss"])
        max1 = max(history.history["loss"])
        plt.semilogx(history.history["lr"], history.history["loss"])
        plt.axis([lr0, lr0 * 10**(nof_epochs / 30), 0, max1])
        plt.show()

    @staticmethod
    def train(model, feature_list, targets, lr=1e-5, lr_plot=True, tf_batching=False):
        optimizer = keras.optimizers.SGD(lr=lr, momentum=0.9)
        #optimizer = keras.optimizers.Adam(lr=lr)
        model.compile(loss=keras.losses.MeanSquaredError(),
                      optimizer=optimizer,
                      metrics=["mse"])
        early_stopping = keras.callbacks.EarlyStopping(patience=50)
        model_checkpoint = keras.callbacks.ModelCheckpoint("my_checkpoint", save_best_only=True)
        if tf_batching:
            history = model.fit(feature_list, validation_data=targets, batch_size=16, epochs=300, verbose=2,
                                shuffle=False,
                                callbacks=[early_stopping, model_checkpoint])
        else:
            history = model.fit(x=feature_list, y=targets, batch_size=16, validation_split=0.2, epochs=500, verbose=2, shuffle=True,
                  callbacks=[early_stopping])
        #model = keras.models.load_model("my_checkpoint")
        #model.save('./Models/BTCUSDT_05.h5')
        #model = keras.models.load_model("my_checkpoint")
        if lr_plot:
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('Model loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Test'], loc='upper left')
            plt.show()
        return model

    @staticmethod
    def forecast(model, feature_list, targets, split_time, window_size, data_normaliser, batch_size, multi_step=False):
        #valid_list = []
        #for feature_group in feature_list:
        #    valid_list.append(feature_group[split_time:-1])
        #valid_target = targets[split_time:-1]
        time_valid = [x for x in range(window_size, len(targets) + window_size)]
        if multi_step:
            forecast_norm, forecast_multi = BasicMethodsDL.model_forecast(model, feature_list, batch_size=batch_size, multi_step=True)
            return data_normaliser.inverse_transform(forecast_norm), data_normaliser.inverse_transform(forecast_multi)
        else:
            forecast_norm = BasicMethodsDL.model_forecast(model, feature_list, batch_size=batch_size)
            return data_normaliser.inverse_transform(forecast_norm)
        #plt.figure(figsize=(10, 6))
        #BC.plot_series(time_valid, data_normaliser.inverse_transform(targets))
        #BC.plot_series(time_valid, data_normaliser.inverse_transform(forecast_norm))
        #plt.show()
        #print(keras.metrics.mean_absolute_error(targets, forecast_norm).numpy())
        #return data_normaliser.inverse_transform(forecast_norm)

    @staticmethod
    def forecast_t(model, feature_list, targets, batch_size, multi_step=False):
        forecast_norm = BasicMethodsDL.model_forecast(model, feature_list, batch_size=batch_size)
        plt.figure(figsize=(10, 6))
        plt.plot(targets)
        plt.plot(forecast_norm)
        plt.show()
        return BasicMethodsDL.model_forecast(model, feature_list, batch_size=batch_size)


    @staticmethod
    def forecast_full(model, series, series_norm, time, window_size, full_data_transform, batch_size):
        forecast = BasicMethodsDL.model_forecast(model, series_norm[:-1], window_size, batch_size=batch_size)[..., -1, 0]
        #denorm_forecast = full_data_transform.inverse_transform(forecast)[:, 0]
        plt.figure(figsize=(10, 6))
        BasicMethodsDL.plot_series(time, series_norm)
        print(time[window_size:])
        BasicMethodsDL.plot_series(time[window_size:], forecast)
        plt.show()
        #print(keras.metrics.mean_absolute_error(series, forecast).numpy())

    @staticmethod
    def create_model(type, window_size=30):
        if type == 'rnn':
            model = keras.models.Sequential([
                keras.layers.LSTM(100, return_sequences=True, stateful=True, batch_input_shape=[1, window_size, 1]),
                keras.layers.LSTM(100, stateful=True),
                keras.layers.Dense(1),
            ])
            return model
        elif type == 'dense':
            model = keras.models.Sequential()
            model.add(
                keras.layers.Dense(10, activation="relu", input_shape=[window_size]),
                keras.layers.Dense(10, activation="relu"),
                keras.layers.Dense(1))
            return model
        elif type == 'wave_net':
            model = keras.models.Sequential()
            model.add(keras.layers.InputLayer(input_shape=[None, 12]))
            for dilation_rate in (1, 2, 4):
                model.add(
                    keras.layers.Conv1D(filters=32,
                                        kernel_size=2,
                                        strides=1,
                                        dilation_rate=dilation_rate,
                                        padding="causal",
                                        activation="relu")
                )
            model.add(keras.layers.Conv1D(filters=1, kernel_size=1))
            return model
