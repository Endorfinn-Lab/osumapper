import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os, re
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Activation, Dropout, Flatten, LSTM, Dense, TimeDistributed, concatenate

root = "mapdata/"
divisor = 4
time_interval = 16

try:
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"
except:
    pass

def read_npz(fn):
    with np.load(fn) as data:
        wav_data = data["wav"]
        wav_data = np.swapaxes(wav_data, 2, 3)
        train_data = wav_data
        div_source = data["lst"][:, 0]
        div_source2 = data["lst"][:, 11:14]
        div_data = np.concatenate([divisor_array(div_source), div_source2], axis=1)
        lst_data = data["lst"][:, 2:10]
        lst_data = 2 * lst_data - 1
        train_labels = lst_data
    return train_data, div_data, train_labels

def divisor_array(t):
    d_range = list(range(0, divisor))
    return np.array([[int(k % divisor == d) for d in d_range] for k in t])

def read_npz_list():
    npz_list = []
    for file in os.listdir(root):
        if file.endswith(".npz"):
            npz_list.append(os.path.join(root, file))
    return npz_list

def prefilter_data(train_data_unfiltered, div_data_unfiltered, train_labels_unfiltered):
    non_object_end_indices = [i for i,k in enumerate(train_labels_unfiltered) if True or k[4] == -1 and k[5] == -1]
    train_data = train_data_unfiltered[non_object_end_indices]
    div_data = div_data_unfiltered[non_object_end_indices]
    train_labels = train_labels_unfiltered[non_object_end_indices][:, [0, 1, 2, 3, 4]]
    return train_data, div_data, train_labels

def preprocess_npzs(train_data_unfiltered, div_data_unfiltered, train_labels_unfiltered):
    train_data, div_data, train_labels = prefilter_data(train_data_unfiltered, div_data_unfiltered, train_labels_unfiltered)
    if train_data.shape[0]%time_interval > 0:
        train_data = train_data[:-(train_data.shape[0]%time_interval)]
        div_data = div_data[:-(div_data.shape[0]%time_interval)]
        train_labels = train_labels[:-(train_labels.shape[0]%time_interval)]
    train_data2 = np.reshape(train_data, (-1, time_interval, train_data.shape[1], train_data.shape[2], train_data.shape[3]))
    div_data2 = np.reshape(div_data, (-1, time_interval, div_data.shape[1]))
    train_labels2 = np.reshape(train_labels, (-1, time_interval, train_labels.shape[1]))
    return train_data2, div_data2, train_labels2

def get_data_shape():
    for file in os.listdir(root):
        if file.endswith(".npz"):
            train_data_unfiltered, div_data_unfiltered, train_labels_unfiltered = read_npz(os.path.join(root, file))
            train_data, div_data, train_labels = prefilter_data(train_data_unfiltered, div_data_unfiltered, train_labels_unfiltered)
            if train_data.shape[0] == 0:
                continue
            return train_data.shape, div_data.shape, train_labels.shape
    print("cannot find npz!! using default shape")
    return (-1, 7, 32, 2), (-1, 3 + divisor), (-1, 5)

def read_some_npzs_and_preprocess(npz_list):
    train_shape, div_shape, label_shape = get_data_shape()
    td_list = []
    dd_list = []
    tl_list = []
    for fp in npz_list:
        if fp.endswith(".npz"):
            _td, _dd, _tl = read_npz(fp)
            if _td.shape[1:] != train_shape[1:]:
                print("Warning: something wrong found in {}! shape = {}".format(fp, _td.shape))
                continue
            td_list.append(_td)
            dd_list.append(_dd)
            tl_list.append(_tl)
    train_data_unfiltered = np.concatenate(td_list)
    div_data_unfiltered = np.concatenate(dd_list)
    train_labels_unfiltered = np.concatenate(tl_list)
    train_data2, div_data2, train_labels2 = preprocess_npzs(train_data_unfiltered, div_data_unfiltered, train_labels_unfiltered)
    return train_data2, div_data2, train_labels2

def train_test_split(train_data2, div_data2, train_labels2, test_split_count=233):
    new_train_data = train_data2[:-test_split_count]
    new_div_data = div_data2[:-test_split_count]
    new_train_labels = train_labels2[:-test_split_count]
    test_data = train_data2[-test_split_count:]
    test_div_data = div_data2[-test_split_count:]
    test_labels = train_labels2[-test_split_count:]
    return (new_train_data, new_div_data, new_train_labels), (test_data, test_div_data, test_labels)

def set_param_fallback(PARAMS):
    global divisor
    try:
        divisor = PARAMS["divisor"]
    except:
        divisor = 4
    if "train_epochs" not in PARAMS:
        PARAMS["train_epochs"] = 16
    if "train_epochs_many_maps" not in PARAMS:
        PARAMS["train_epochs_many_maps"] = 6
    if "too_many_maps_threshold" not in PARAMS:
        PARAMS["too_many_maps_threshold"] = 200
    if "data_split_count" not in PARAMS:
        PARAMS["data_split_count"] = 80
    if "plot_history" not in PARAMS:
        PARAMS["plot_history"] = True
    if "train_batch_size" not in PARAMS:
        PARAMS["train_batch_size"] = None
    return PARAMS

def build_model():
    train_shape, div_shape, label_shape = get_data_shape()
    wav_input = Input(shape=(time_interval, train_shape[1], train_shape[2], train_shape[3]), name='wav_input')
    x1 = TimeDistributed(Conv2D(16, (2, 2), data_format='channels_last'))(wav_input)
    x1 = TimeDistributed(MaxPool2D((1, 2), data_format='channels_last'))(x1)
    x1 = TimeDistributed(Activation(tf.nn.relu))(x1)
    x1 = TimeDistributed(Dropout(0.3))(x1)
    x1 = TimeDistributed(Conv2D(16, (2, 3), data_format='channels_last'))(x1)
    x1 = TimeDistributed(MaxPool2D((1, 2), data_format='channels_last'))(x1)
    x1 = TimeDistributed(Activation(tf.nn.relu))(x1)
    x1 = TimeDistributed(Dropout(0.3))(x1)
    x1 = TimeDistributed(Flatten())(x1)
    lstm_out = LSTM(64, activation=tf.nn.tanh, return_sequences=True)(x1)
    div_input = Input(shape=(time_interval, div_shape[1]), name='div_input')
    conc = concatenate([lstm_out, div_input])
    dense1 = Dense(71, activation=tf.nn.tanh)(conc)
    dense2 = Dense(71, activation=tf.nn.relu)(dense1)
    final_output = Dense(label_shape[1], activation=tf.nn.tanh)(dense2)
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    final_model = Model(inputs=[wav_input, div_input], outputs=final_output)
    final_model.compile(loss='mse',
                        optimizer=optimizer,
                        metrics=[keras.metrics.mae])
    return final_model

def plot_history(history):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [Limitless]')
    plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
           label='Train MAE')
    plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
           label = 'Val MAE')
    plt.plot(history.epoch, np.array(history.history['loss']),
           label='Train Loss')
    plt.plot(history.epoch, np.array(history.history['val_loss']),
           label = 'Val Loss')
    plt.legend()
    plt.show()

class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs):
        if epoch % 100 == 0: print('')
        print('.', end='')

def step2_build_model():
    model_v7 = build_model()
    return model_v7

def step2_train_model(model, PARAMS):
    global history, new_train_data, new_div_data, new_train_labels, test_data, test_div_data, test_labels
    PARAMS = set_param_fallback(PARAMS)
    train_file_list = read_npz_list()
    EPOCHS = PARAMS["train_epochs"]
    too_many_maps_threshold = PARAMS["too_many_maps_threshold"]
    data_split_count = PARAMS["data_split_count"]
    batch_size = PARAMS["train_batch_size"]
    early_stop = keras.callbacks.EarlyStopping(monitor='mean_absolute_error', patience=20)
    if len(train_file_list) >= too_many_maps_threshold:
        EPOCHS = PARAMS["train_epochs_many_maps"]
    if len(train_file_list) < too_many_maps_threshold:
        train_data2, div_data2, train_labels2 = read_some_npzs_and_preprocess(train_file_list)
        (new_train_data, new_div_data, new_train_labels), (test_data, test_div_data, test_labels) = train_test_split(train_data2, div_data2, train_labels2)
        history = model.fit([new_train_data, new_div_data], new_train_labels, epochs=EPOCHS,
                            validation_split=0.2, verbose=0, batch_size=batch_size,
                            callbacks=[early_stop, PrintDot()])
        if PARAMS["plot_history"]:
            plot_history(history)
    else:
        for epoch in range(EPOCHS):
            print(f"Epoch {epoch+1}/{EPOCHS}")
            for map_batch in range(np.ceil(len(train_file_list) / data_split_count).astype(int)):
                if map_batch == 0:
                    train_data2, div_data2, train_labels2 = read_some_npzs_and_preprocess(train_file_list[map_batch * data_split_count : (map_batch+1) * data_split_count])
                    (new_train_data, new_div_data, new_train_labels), (test_data, test_div_data, test_labels) = train_test_split(train_data2, div_data2, train_labels2)
                else:
                    new_train_data, new_div_data, new_train_labels = read_some_npzs_and_preprocess(train_file_list[map_batch * data_split_count : (map_batch+1) * data_split_count])
                history = model.fit([new_train_data, new_div_data], new_train_labels, epochs=1,
                                    validation_data=([test_data, test_div_data], test_labels),
                                    verbose=0, batch_size=batch_size,
                                    callbacks=[])
                print('.', end='')
            print('')
    return model

from sklearn.metrics import roc_auc_score

def step2_evaluate(model):
    try:
        if 'test_data' not in globals() or test_data.size == 0:
            print("Test data not available for evaluation. Please run training first.")
            return
    except NameError:
        print("Test data not available for evaluation. Please run training first.")
        return
        
    train_shape, div_shape, label_shape = get_data_shape()
    test_predictions = model.predict([test_data, test_div_data])
    flat_test_preds = test_predictions.reshape(-1, label_shape[1])
    flat_test_labels = test_labels.reshape(-1, label_shape[1])
    pred_result = (flat_test_preds + 1) / 2
    actual_result = (flat_test_labels + 1) / 2
    column_names = ["is_note_start", "is_circle", "is_slider", "is_spinner", "is_note_end"]
    for i, k in enumerate(column_names):
        if len(np.unique(actual_result[:, i])) < 2:
            print(f"Skipping AUC for '{k}': only one class present in y_true.")
            continue
        if i == 3:
            continue
        if i == 2 and np.sum(actual_result[:, i]) == 0:
            continue
        print("{} auc score: {}".format(k, roc_auc_score(actual_result[:, i], pred_result[:, i])))

def step2_save(model):
    tf.keras.models.save_model(
        model,
        "saved_rhythm_model.h5",
        overwrite=True,
        include_optimizer=True
    )
    print("\nModel saved as saved_rhythm_model.h5")