import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn import preprocessing
from sklearn import metrics
import matplotlib.pyplot as plt
tf.random.set_seed(123)
np.random.seed(123)

df = pd.read_csv('#dataset_path')
df.head()
df = df[['#all the variables in the dataset including the visibility']]
validate = df[['#mention all the visibility related variables']].tail(#mention the number of rows to be shown)
df.drop(df.tail(#mention the number of rows to be shown).index,inplace=True)

x_scaler = preprocessing.MinMaxScaler()
y_scaler = preprocessing.MinMaxScaler()
dataX = x_scaler.fit_transform(df[['#mention all the datasets including the visibility']] )
dataY = y_scaler.fit_transform(df[['VISI']])

def custom_ts_multi_data_prep(dataset, target, start, end, window, horizon):
    X = []
    y = []
    start = start + window
    if end is None:
        end = len(dataset) - horizon

    for i in range(start, end):
        indices = range(i-window, i)
        X.append(dataset[indices])

        indicey = range(i+1, i+1+horizon)
        y.append(target[indicey])
    return np.array(X), np.array(y)

hist_window = #mention the look back window size
horizon = #number of data points to be forecasted
TRAINSPLIT = int(len(df)*0.8)
x_train_multi, y_train_multi = custom_ts_multi_data_prep(
    dataX, dataY, 0, TRAINSPLIT, hist_window, horizon)
x_val_multi, y_val_multi = custom_ts_multi_data_prep(
    dataX, dataY, TRAINSPLIT, None, hist_window, horizon)

BATCH_SIZE = #mention the batch size
BUFFER_SIZE = #mention the buffer size

train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()

Bi_lstm_model = tf.keras.models.Sequential([
  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(#mention the units, return_sequences=True), 
                               input_shape=x_train_multi.shape[-2:]),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(#mention the layers)),
    tf.keras.layers.Dense(#mention the dense layers, activation='relu'),
    tf.keras.layers.Dropout(#dropout percentage),
    tf.keras.layers.Dense(units=horizon),
])
Bi_lstm_model.compile(optimizer='adam', loss='mse')

model_path = r'model_path'

EVALUATION_INTERVAL = #mention the evaluation interval
EPOCHS = #mention the epoch
history = Bi_lstm_model.fit(train_data_multi, epochs=EPOCHS, steps_per_epoch=EVALUATION_INTERVAL, validation_data=val_data_multi, validation_steps=#, verbose=1,
                                   callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='min'),tf.keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, mode='min', verbose=0)])


Trained_model = tf.keras.models.load_model(model_path)

# Model architecture
Trained_model.summary()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train loss', 'validation loss'], loc='upper left')
plt.rcParams["figure.figsize"] = [16,9]
plt.show()

data_val = x_scaler.fit_transform(df[['#all the variables in the dataset']].tail(#number of rows to be specified))

val_rescaled = data_val.reshape(1, data_val.shape[0], data_val.shape[1])

Predicted_results = Trained_model.predict(val_rescaled)

print(Predicted_results)
Predicted_results_Inv_trans = y_scaler.inverse_transform(Predicted_results)

print(Predicted_results_Inv_trans)


def timeseries_evaluation_metrics_func(y_true, y_pred):
    
    def mean_absolute_percentage_error(y_true, y_pred): 
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print('Evaluation metric results:-')
    print(f'MSE is : {metrics.mean_squared_error(y_true, y_pred)}')
    print(f'MAE is : {metrics.mean_absolute_error(y_true, y_pred)}')
    print(f'RMSE is : {np.sqrt(metrics.mean_squared_error(y_true, y_pred))}')
    print(f'MAPE is : {mean_absolute_percentage_error(y_true, y_pred)}')
    print(f'R2 is : {metrics.r2_score(y_true, y_pred)}',end='\n\n')

timeseries_evaluation_metrics_func(validate['VISIBILITY'],Predicted_results_Inv_trans[0])

plt.plot( list(validate['VISIBILITY']))
plt.plot( list(Predicted_results_Inv_trans[0]))
plt.title("Actual vs Predicted")
plt.ylabel("VISIBILITY")
plt.xlabel("number of hours")
plt.legend(('Actual','predicted'))
plt.show()

