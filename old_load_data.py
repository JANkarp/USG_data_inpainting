import pickle
import numpy as np
import glob
import os
import torch
from scipy.signal import butter, sosfiltfilt
from sklearn import model_selection

#Used for loading data for the old U-net pipeline,
#wouldn't work on bigger dataset used by new pipeline due to memory constraints
#can return both the rf data and iq data representations

def load(path, param, mode, decimate):
    Y, freq_list = read_data(path)

    #Podział na zbiory treningowy/walidacyjny/testowy - 80/10/10
    train_Y_raw, test_Y_raw, train_freq, test_freq = model_selection.train_test_split(Y, freq_list, test_size=0.1, random_state=42)
    train_Y_raw, val_Y_raw, train_freq, val_freq = model_selection.train_test_split(train_Y_raw, train_freq, test_size=1/9, random_state=42)


    #normalizacja
    global_mean = np.mean(train_Y_raw)
    global_std = np.std(train_Y_raw)


    train_Y = (train_Y_raw - global_mean) / global_std
    val_Y = (val_Y_raw - global_mean) / global_std
    test_Y = (test_Y_raw - global_mean) / global_std


    #Stworzenie danych wejściowych
    train_X = insert_zeros(train_Y,param)
    val_X = insert_zeros(val_Y,param)
    test_X = insert_zeros(test_Y,param)
    extended_X = get_extended(test_Y,param)

    if mode == 'rf':
        train_x_tensor = convert_to_tensor(train_X)
        train_y_tensor = convert_to_tensor(train_Y)
        val_x_tensor = convert_to_tensor(val_X)
        val_y_tensor = convert_to_tensor(val_Y)
        test_x_tensor = convert_to_tensor(test_X)
        test_y_tensor = convert_to_tensor(test_Y)
        extended_x_tensor = convert_to_tensor(extended_X)

        return train_x_tensor, val_x_tensor, test_x_tensor, train_y_tensor, val_y_tensor, test_y_tensor, extended_x_tensor, test_freq, global_mean, global_std

    else:
        #Zamiana na tensory
        complex_train_X = convert_to_iq(train_X, train_freq, decimate)
        complex_train_Y = convert_to_iq(train_Y, train_freq, decimate)
        complex_val_X = convert_to_iq(val_X, val_freq, decimate)
        complex_val_Y = convert_to_iq(val_Y, val_freq, decimate)
        complex_test_X = convert_to_iq(test_X, test_freq, decimate)
        complex_test_Y = convert_to_iq(test_Y, test_freq, decimate)
        complex_extended_X = convert_to_iq(extended_X, test_freq, decimate)

        train_x_tensor = convert_to_tensor(complex_train_X)
        train_y_tensor = convert_to_tensor(complex_train_Y)
        val_x_tensor = convert_to_tensor(complex_val_X)
        val_y_tensor = convert_to_tensor(complex_val_Y)
        test_x_tensor = convert_to_tensor(complex_test_X)
        test_y_tensor = convert_to_tensor(complex_test_Y)
        extended_x_tensor = convert_to_tensor(complex_extended_X)

        return train_x_tensor, val_x_tensor, test_x_tensor, train_y_tensor, val_y_tensor, test_y_tensor, extended_x_tensor, test_freq,  global_mean, global_std

def insert_zeros(Y, param):
    X = np.zeros_like(Y)
    for frame_idx in range(X.shape[0]):
        for idx in range(0,X.shape[2],param):
            X[frame_idx, :, idx] = Y[frame_idx, :, idx]
    return X

def get_extended(Y, param):
    extended_X = np.zeros((Y.shape[0], Y.shape[1], Y.shape[2]*param))

    for frame_idx in range(Y.shape[0]):
        for idx in range(0,extended_X.shape[2],param):
            extended_X[frame_idx, :, idx] = Y[frame_idx, :, int(idx/param)]

    return extended_X

def convert_to_iq(data, freq, decimate):
    fs = 65 * 1e6
    H = data.shape[1]
    t = np.arange(H) / fs

    f_block = freq[:, np.newaxis, np.newaxis]
    t_block = t[np.newaxis, :, np.newaxis]

    data_sin = data * np.sin(2 * np.pi * f_block * t_block)
    data_cos = data * np.cos(2 * np.pi * f_block * t_block)

    used_freqs = set(freq)

    filtered_data_sin = np.zeros_like(data)
    filtered_data_cos = np.zeros_like(data)

    for u_freq in used_freqs:
        idx = np.where(freq == u_freq)
        Wn = 0.8 * u_freq
        sos = butter(5, Wn, 'lowpass', analog=False, fs=fs, output='sos')
        filtered_data_sin[idx] = 2 * sosfiltfilt(sos, data_sin[idx], axis = 1)
        filtered_data_cos[idx] = 2 * sosfiltfilt(sos, data_cos[idx], axis = 1)

    decimated_i = filtered_data_sin[:,0::decimate,:]
    decimated_j = filtered_data_cos[:,0::decimate,:]

    iq_data = decimated_j + 1j * decimated_i

    return iq_data

def convert_to_tensor(data):
    if np.iscomplexobj(data):
        i_channel = np.real(data)
        q_channel = np.imag(data)

        iq_data = np.stack([i_channel, q_channel], axis=1)

        return torch.tensor(iq_data, dtype=torch.float32)
    else:
        output = torch.tensor(data, dtype=torch.float32)
        return output.unsqueeze(1)

def read_data(path):
    csv_files = glob.glob(os.path.join(path, '*.pkl'))
    arrays_list = []
    freq_list = []
    for file in csv_files:
        filename = os.path.basename(file)
        with open(file, 'rb') as f:

            data = pickle.load(f)

            tuples = data['data']
            for letter in filename:
                if letter.isdigit():
                    for _ in range(len(tuples)):
                        freq_list.append(int(letter) * 1e6)
                    break
            for tup in tuples:
                arrays_list.append(tup[0])

    Y = np.stack(arrays_list, axis=0)
    freq_list = np.array(freq_list)
    print(f"Loaded {len(arrays_list)} frames.")
    print(f"Final data shape: {Y.shape}")

    return Y, freq_list

def get_db(frame):
    mag = np.abs(frame)
    mag_norm = mag / np.max(mag)
    log_data = 20 * np.log10(mag_norm + 1e-5)
    return log_data

def print_meta(metas):
    print("Available Metadata Attributes:")
    for meta in metas:
        print([attr for attr in dir(meta) if not attr.startswith('_')])


        if hasattr(meta, 'context'):
            ctx = meta.context
            print([attr for attr in dir(ctx) if not attr.startswith('_')])

            if hasattr(ctx, 'sequence'):
                seq = ctx.sequence
                print("\n=== Transmit Sequence ===")
                print("Plane Wave Angles (rad):", getattr(seq, 'angles', 'N/A'))
                print("Speed of Sound (m/s):", getattr(seq, 'speed_of_sound', 'N/A'))


        if hasattr(meta, 'data_description'):
            print("\n=== Data Description ===")
            print("Sampling Frequency / Dtype / Shape:", meta.data_description)

        if hasattr(meta, 'dtype'):
            print("\n=== Data Type ===")
            print("data type:", meta.dtype)

        if hasattr(meta, 'input_shape'):
            print("\n=== Input Shape ===")
            print( meta.input_shape)

        if hasattr(meta, 'is_iq_data'):
            print("\n=== IQ? ===")
            print( meta.is_iq_data)