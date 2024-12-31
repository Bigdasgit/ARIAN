import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def data_load(data, node_num, input_len, output_len):
    scaler_speed = StandardScaler()
    scaler_volume = StandardScaler()
    data_np = np.load(data)
    data_np = data_np['data'].astype('float32')

    print(data_np.shape)
    if data_np.ndim == 2:
        data_np = np.expand_dims(data_np, axis=2)
        print(data_np.shape)
    
    speed_data = data_np[:, node_num*(12-input_len):node_num*12, 0]
    volume_data = data_np[:, node_num*(12-input_len):node_num*12, 1]
    
    # Std for speed
    scaler_speed.fit(speed_data)
    std_data_speed = scaler_speed.transform(speed_data)

    # Std for volume
    scaler_volume.fit(volume_data)
    std_data_volume = scaler_volume.transform(volume_data)

    # Concate speed and volume >> Input sequence
    std_data_speed = np.expand_dims(std_data_speed, axis=2)
    std_data_volume = np.expand_dims(std_data_volume, axis=2)
    out_np = np.concatenate((std_data_speed, std_data_volume), axis=2)

    # Concate intput sequence and output sequence
    out_np = np.concatenate((out_np, data_np[:, :node_num*output_len, 2:3]), axis=2)
    print(out_np.shape)
    
    return out_np, scaler_speed


# def data_load(data, node_num, input_len, output_len):
#     scaler = StandardScaler()
#     data_np = np.load(data)
#     data_np = data_np['data'].astype('float32')
#     print(data_np.shape)
#     if data_np.ndim == 2:
#         data_np = np.expand_dims(data_np, axis=2)
#         print(data_np.shape)
#     # data_np = data_np.reshape(-1, node_num, 2)
#     scaler.fit(data_np[:, node_num*(12-input_len):node_num*12, 0])
#     print(scaler)
#     std_data = scaler.transform(data_np[:, node_num*(12-input_len):node_num*12, 0])
#     out_np = np.concatenate(std_data, data_np[:, :node_num*output_len, 1], axis=1)
#     print(out_np.shape)
#     # data_np = data_np.reshape(-1, node_num*12, 2)

#     return out_np, scaler

def data_load_(data, node_num):
    scaler = StandardScaler()
    # data_np = np.load('C:/Users/joker/Desktop/newidea/data/' + data)                    #########################
    data_np = np.load(data)    
    data_np = data_np['data']
    print(data_np.shape)
    # data_np = data_np.reshape(-1, node_num, 2)
    scaler.fit(data_np[:, :, 0])
    scaler.partial_fit(data_np[:, :, 1])
    print(scaler)
    data_np[:, :, 0] = scaler.transform(data_np[:, :, 0])
    data_np[:, :, 1] = scaler.transform(data_np[:, :, 1])
    print(data_np.shape)
    # data_np = data_np.reshape(-1, node_num*12, 2)

    return data_np, scaler


