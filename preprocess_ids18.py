'''
clean and save compressed data. Also sort by timestamps, ascending order.

nohup python -u preprocess_ids18.py > logs/preprocess_ids18.log &

clean strategy:
Traffic with NaN and Infinity values was removed.
Duplicate traffic was removed (duplicate means exactly same traffic).

For "02_20_2018.csv" file, it has 4 extra features in the beninning (Flow ID, Src IP, Src Port, Dst IP,)
-> extra features were removed.

developed deeply based on CADE
'''

import os, sys
import traceback
import numpy as np


from datetime import datetime
from timeit import default_timer as timer
from collections import Counter
from tqdm import tqdm
from pprint import pformat
from timeit import default_timer as timer

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split



# put origin csv files in here
RAW_DATA_PATH = "./data/CICIDS18/origindata"
# =================================================

UNNORMALIZED_SAVE_FOLDER = './data/CICIDS18/unnormalized'
NOISEDATA_SAVE_FOLDER = './data/CICIDS18/origindata'

SAVE_PATH = "/home/maybo/Documents/Project/Paper/etc-fusion-py/Tasks/draft-RLCor/dataset"

NORMAL_FILES = ['Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv', 'Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv', 'Friday-16-02-2018_TrafficForML_CICFlowMeter.csv', 'Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv',
                'Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv', 'Friday-23-02-2018_TrafficForML_CICFlowMeter.csv', 'Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv', 'Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv', 'Friday-02-03-2018_TrafficForML_CICFlowMeter.csv']

NORMAL_MAL_FILES = ['Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv', 'Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv', 'Friday-16-02-2018_TrafficForML_CICFlowMeter.csv', 'Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv',
                'Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv', 'Friday-23-02-2018_TrafficForML_CICFlowMeter.csv', 'Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv', 'Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv', 'Friday-02-03-2018_TrafficForML_CICFlowMeter.csv']

NORMAL_NEG_FILES = ['Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv']


SPECIFIC_FILES = ['Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv']

TRAFFIC_LABEL = {'Benign': 0,
                 'FTP-BruteForce': 1, 'SSH-Bruteforce': 1,
                 'DoS attacks-GoldenEye': 2, 'DoS attacks-Slowloris': 2,
                 'DoS attacks-SlowHTTPTest': 2, 'DoS attacks-Hulk': 2,
                 'Infilteration': 3,
                 'DDoS attacks-LOIC-HTTP': 4, 'DDOS attack-LOIC-UDP': 4, 'DDOS attack-HOIC': 4,
                 'Brute Force -Web': 5,
                 'Brute Force -XSS': 5,
                 'Bot': 6,
                 'SQL Injection': 7
                 }




def main():

    # ===================================================
    # clean csv files
    start = timer()
    for file in NORMAL_FILES:
        file_path = os.path.join(RAW_DATA_PATH, file)
        clean_single_file(file_path)

    s1 = timer()
    for file in SPECIFIC_FILES:
        file_path = os.path.join(RAW_DATA_PATH, file)
        for i in range(1, 5) :
            clean_single_file(file_path, is_specific=True, idxii= i)
    e1 = timer()
    print(f'biggest file (4G): {e1 - s1}')

    end = timer()
    print(f'time elapsed: {end - start}')

    '''calc the # of each label in each file'''
    stats()

    # ===================================
    ''' parse required training and testing files, concatenate and resplit them. '''
    # 划分恶意数据并保存
    saved_unnormalized_path = os.path.join(UNNORMALIZED_SAVE_FOLDER, f'mal_all_e2002_unnormalized.npz')
    split_mal_data(NORMAL_MAL_FILES, saved_unnormalized_path)
    saved_unnormalized_path = os.path.join(UNNORMALIZED_SAVE_FOLDER, f'mal_all_2002_unnormalized.npz')
    split_mal_data(SPECIFIC_FILES, saved_unnormalized_path)
    # 合并数据集并保存
    saved_unnormalized_path = os.path.join(UNNORMALIZED_SAVE_FOLDER, f'mal_all_unnormalized.npz')
    con_split_mal_data(saved_unnormalized_path)

    # 划分良性数据并保存
    saved_unnormalized_path = os.path.join(UNNORMALIZED_SAVE_FOLDER, f'neg_all_2202_unnormalized.npz')
    split_neg_data(NORMAL_NEG_FILES, saved_unnormalized_path)

    # 合并数据集并保存
    saved_unnormalized_path = os.path.join(UNNORMALIZED_SAVE_FOLDER, f'all_unnormalized.npz')
    con_split_data(saved_unnormalized_path)

    # ''' normalize train, test and save them to file. '''
    data_file_path = os.path.join(UNNORMALIZED_SAVE_FOLDER, 'all_unnormalized.npz')
    raw_data = np.load(data_file_path)
    X_train, y_train,X_test ,y_test  = raw_data['X_train'], raw_data['y_train'], raw_data['X_test'], raw_data['y_test']
    save_path = os.path.join(NOISEDATA_SAVE_FOLDER, 'noise_data.npz')
    normalize(X_train, X_test, y_train, y_test, 0.1, save_path)

    # ===========================================================
    # 全随机筛选出来元数据集， 比例为0.01
    random_getmete(NOISEDATA_SAVE_FOLDER, 'meta_data_random_001.npz', 0.01)
    # 按类别，固定比例和数量筛选出来的平衡元数据集, 数量为500， 100， 50
    for i in [500,100,50]:
        save_name = 'meta_data_balance_' + str(i) + '.npz'
        balance_getmete(NOISEDATA_SAVE_FOLDER, save_name, i)

    # =============================================================
    noise_ratio = [.1, .3, .5, .7]
    # asy
    set_noise_label_asy(noise_ratio, NOISEDATA_SAVE_FOLDER, 'noise_asy_labels.npz')
    # sy
    set_noise_label_sy(noise_ratio, NOISEDATA_SAVE_FOLDER, 'noise_sy_labels.npz')
    # unannotated 将要求的某类完全平均给其他类
    set_noise_label_ask(NOISEDATA_SAVE_FOLDER, 'noise_ask_labels.npz')





def clean_single_file(filename, is_specific=False, idxii=1):
    traffic_contain_null_count = 0
    traffic_contain_infinity_count = 0
    traffic_invalid_timestamp_count = 0

    traffics = [] # a list of traffics, including feature vector and label names.

    print(f'cleaning file {filename}...')

    '''remove traffic with NaN and Infinity values, read the file content into a numpy array.'''
    with open(filename, 'r') as f:
        date_str = filename.replace('_TrafficForML_CICFlowMeter.csv', '').replace(RAW_DATA_PATH, '').replace('-', '/')
        date_str = date_str.replace(date_str.split('/')[1], '')
        date_str = date_str[2:]
        print(f'date_str: {date_str}')
        next(f)
        contents = f.readlines()
        for idx, line in enumerate(contents):
            try:

                if idx < (idxii -1) * 4000000 :
                    continue
                if idx > (idxii ) * 4000000 :
                    break

                line = line.strip().split(',')

                if is_specific:
                    # the 02_20_2018 file has 4 extra features in the beginning
                    line = line[4:]
                if not line[0].isdigit():  # useless traffic, which is not a feature vector of traffic
                    continue
                if 'NaN' in line:
                    traffic_contain_null_count += 1
                    continue
                if 'Infinity' in line:
                    traffic_contain_infinity_count += 1
                    continue
                if date_str not in line[2]:
                    traffic_invalid_timestamp_count += 1
                    continue
                # convert datetime to UNIX timestamp
                line[2] = str(datetime.strptime(line[2], '%d/%m/%Y %H:%M:%S').timestamp())
                traffics.append(line)
            except:
                print(f'{filename} line {idx} error')
                print(traceback.format_exc())
    print('===================\n')
    print(f'total # traffic is {len(contents)}')
    print(f'traffic that has NaN values: {traffic_contain_null_count}')
    print(f'traffic that has Infinity values: {traffic_contain_infinity_count}')
    print(f'traffic that has invalid timestamp: {traffic_invalid_timestamp_count}')

    traffics = np.array(traffics)
    print(f'after removing NaN, Infinity, and invalid, traffic shape: {traffics.shape}')

    '''remove duplicate traffics and save feature vectors, assigned labels, semantic labels to a compressed file.'''
    # np.unique will sort instead of keeping the original order.
    unique_traffics = np.unique(traffics, axis=0)
    sorted_traffics = unique_traffics[np.argsort(unique_traffics[:, 2])]  # sort by the timestamp of the traffic

    X = sorted_traffics[:, 0:-1].astype(np.float)  # feature vectors
    y_name = sorted_traffics[:, -1] # full name indicating the meaning of the label.

    y = [] # assigned labels
    for name in y_name:
        y.append(TRAFFIC_LABEL[name]) # convert label name to number
    y = np.array(y)

    base_filename = os.path.basename(filename)
    save_file_path = os.path.join(RAW_DATA_PATH, base_filename.replace('csv', str(idxii) + '.npz'))

    np.savez_compressed(save_file_path, X=X, y=y, y_name=y_name)
    print(f'sorted traffics shape: {X.shape}')
    print(f'labels shape: {y.shape}')
    print(f'label names shape: {y_name.shape}')
    print(f'percentage of kept traffic (removing NaN, Infinity, duplicates): {X.shape[0] / len(contents)}')
    print('===================\n')


def stats():
    for file in NORMAL_FILES + SPECIFIC_FILES:
        print(f'stats for file: {file}')
        file_path = os.path.join(SAVE_PATH, file + '.npz')
        data = np.load(file_path)
        y, y_name = data['y'], data['y_name']
        assigned_labels = Counter(label for label in y)
        semantic_labels = Counter(name for name in y_name)
        print(f'assigned labels: {assigned_labels}')
        print(f'semantic labels: {semantic_labels}')


def split_mal_data(NORMAL_MAL_FILES, saved_unnormalized_path):
    # 将所有的非良性流量全部集合在一起
    # 提取并聚合恶性流量
    # Extract malicious feature vectors and labels
    X_mal_list, y_mal_list = [], []
    for filename in NORMAL_MAL_FILES:
        X_mal, y_mal = extract_mal(filename)
        X_mal_list.append(X_mal)
        y_mal_list.append(y_mal)

    for idx, (X, y) in enumerate(zip(X_mal_list, y_mal_list)):
        X_train_tmp, X_test_tmp, y_train_tmp, y_test_tmp = train_test_split(X, y, test_size=0.2, shuffle=False)
        if idx == 0:
            X_train, X_test, y_train, y_test = X_train_tmp, X_test_tmp, y_train_tmp, y_test_tmp
        else:
            X_train = np.concatenate((X_train, X_train_tmp), axis=0)
            X_test = np.concatenate((X_test, X_test_tmp), axis=0)
            y_train = np.concatenate((y_train, y_train_tmp), axis=0)
            y_test = np.concatenate((y_test, y_test_tmp), axis=0)


    print(f'before X_train: {X_train.shape}, y_train: {y_train.shape}')
    print(f'before X_test: {X_test.shape}, y_test: {y_test.shape}')
    print(f'before y_train Counter: {Counter(y_train)}')
    print(f'before y_test Counter: {Counter(y_test)}')

    np.savez_compressed(saved_unnormalized_path,
                        X_train=X_train, y_train=y_train,
                        X_test=X_test, y_test=y_test)


    return X_train, X_test, y_train, y_test

def extract_mal(filename):
    filename = filename.replace('.csv', '.npz')
    data_file_path = os.path.join(RAW_DATA_PATH, filename)
    raw_data = np.load(data_file_path)
    X, y, y_name = raw_data['X'], raw_data['y'], raw_data['y_name']
    data_filter = np.where(y_name != "Benign")[0]
    data = X[data_filter]
    label = y[data_filter]
    sort_idx = np.argsort(data[:, 2], kind='mergesort')
    sorted_data = data[sort_idx]
    sorted_label = label[sort_idx]
    print(f'sorted_mal_data {filename}.shape: {sorted_data.shape}')
    print(f'sorted_mal_label {filename}.shape: {sorted_label.shape}')
    return sorted_data, sorted_label

def split_neg_data(NORMAL_NEG_FILES, saved_unnormalized_path):
    # Extract malicious feature vectors and labels
    X_mal_list, y_mal_list = [], []
    for filename in NORMAL_NEG_FILES:
        X_mal, y_mal = extract_neg(filename)
        X_mal_list.append(X_mal)
        y_mal_list.append(y_mal)

    for idx, (X, y) in enumerate(zip(X_mal_list, y_mal_list)):
        X_train_tmp, X_test_tmp, y_train_tmp, y_test_tmp = train_test_split(X, y, test_size=0.2, shuffle=False)
        if idx == 0:
            X_train, X_test, y_train, y_test = X_train_tmp, X_test_tmp, y_train_tmp, y_test_tmp
        else:
            X_train = np.concatenate((X_train, X_train_tmp), axis=0)
            X_test = np.concatenate((X_test, X_test_tmp), axis=0)
            y_train = np.concatenate((y_train, y_train_tmp), axis=0)
            y_test = np.concatenate((y_test, y_test_tmp), axis=0)

    print(f'before X_train: {X_train.shape}, y_train: {y_train.shape}')
    print(f'before X_test: {X_test.shape}, y_test: {y_test.shape}')
    print(f'before y_train Counter: {Counter(y_train)}')
    print(f'before y_test Counter: {Counter(y_test)}')

    np.savez_compressed(saved_unnormalized_path,
                        X_train=X_train, y_train=y_train,
                        X_test=X_test, y_test=y_test)

    return X_train, X_test, y_train, y_test

def con_split_mal_data(saved_unnormalized_path):
    data_file_path = os.path.join(UNNORMALIZED_SAVE_FOLDER, 'mal_all_2002_unnormalized.npz')
    raw_data = np.load(data_file_path)
    X_train, y_train,X_test ,y_test  = raw_data['X_train'], raw_data['y_train'], raw_data['X_test'], raw_data['y_test']

    data_file_path = os.path.join(UNNORMALIZED_SAVE_FOLDER, 'mal_all_e2202_unnormalized.npz')
    raw_data = np.load(data_file_path)
    X_train_tmp, y_train_tmp,X_test_tmp ,y_test_tmp  = raw_data['X_train'], raw_data['y_train'], raw_data['X_test'], raw_data['y_test']

    X_train = np.concatenate((X_train, X_train_tmp), axis=0)
    X_test = np.concatenate((X_test, X_test_tmp), axis=0)
    y_train = np.concatenate((y_train, y_train_tmp), axis=0)
    y_test = np.concatenate((y_test, y_test_tmp), axis=0)

    print(f'before X_train: {X_train.shape}, y_train: {y_train.shape}')
    print(f'before X_test: {X_test.shape}, y_test: {y_test.shape}')
    print(f'before y_train Counter: {Counter(y_train)}')
    print(f'before y_test Counter: {Counter(y_test)}')

    np.savez_compressed(saved_unnormalized_path,
                        X_train=X_train, y_train=y_train,
                        X_test=X_test, y_test=y_test)


def con_split_data(saved_unnormalized_path):
    data_file_path = os.path.join(UNNORMALIZED_SAVE_FOLDER, 'mal_all_unnormalized.npz')
    raw_data = np.load(data_file_path)
    X_train, y_train,X_test ,y_test  = raw_data['X_train'], raw_data['y_train'], raw_data['X_test'], raw_data['y_test']

    data_file_path = os.path.join(UNNORMALIZED_SAVE_FOLDER, 'neg_all_2202_unnormalized.npz')
    raw_data = np.load(data_file_path)
    X_train_tmp, y_train_tmp,X_test_tmp ,y_test_tmp  = raw_data['X_train'], raw_data['y_train'], raw_data['X_test'], raw_data['y_test']

    X_train = np.concatenate((X_train, X_train_tmp), axis=0)
    X_test = np.concatenate((X_test, X_test_tmp), axis=0)
    y_train = np.concatenate((y_train, y_train_tmp), axis=0)
    y_test = np.concatenate((y_test, y_test_tmp), axis=0)

    print(f'before X_train: {X_train.shape}, y_train: {y_train.shape}')
    print(f'before X_test: {X_test.shape}, y_test: {y_test.shape}')
    print(f'before y_train Counter: {Counter(y_train)}')
    print(f'before y_test Counter: {Counter(y_test)}')

    np.savez_compressed(saved_unnormalized_path,
                        X_train=X_train, y_train=y_train,
                        X_test=X_test, y_test=y_test)

def extract_neg(filename):
    filename = filename.replace('.csv', '.npz')
    data_file_path = os.path.join(RAW_DATA_PATH, filename)
    raw_data = np.load(data_file_path)
    X, y, y_name = raw_data['X'], raw_data['y'], raw_data['y_name']
    data_filter = np.where(y_name == "Benign")[0]
    data = X[data_filter]
    label = y[data_filter]
    sort_idx = np.argsort(data[:, 2], kind='mergesort')
    sorted_data = data[sort_idx]
    sorted_label = label[sort_idx]
    print(f'sorted_mal_data {filename}.shape: {sorted_data.shape}')
    print(f'sorted_mal_label {filename}.shape: {sorted_label.shape}')
    return sorted_data, sorted_label


def normalize(X_train, X_test, y_train, y_test, ratio, save_path):
    print(f'y_train unique: {np.unique(y_train)}')

    ''' downsampling '''
    X_train, y_train = downsampling(X_train, y_train, ratio, phase='train')
    X_test, y_test = downsampling(X_test, y_test, ratio, phase='test')

    ''' calculate frequency for Dst Port feature in the training set, change the port feature to
    high (0), medium (1), low (2), then one hot encoder to a 3-dimensional array, then fit testing set. '''
    training_ports = X_train[:, 0]
    training_ports_counter = Counter(training_ports)
    print(f'training ports top 20: {training_ports_counter.most_common(20)}')

    high_freq_port_list = []
    medium_freq_port_list = []
    low_freq_port_list = []
    for port in training_ports_counter:
        count = training_ports_counter[port]
        if count >= 10000:
            high_freq_port_list.append(port)
        elif count >= 1000:
            medium_freq_port_list.append(port)
        else:
            low_freq_port_list.append(port)

    training_ports_transform = transform_ports_to_categorical(training_ports, high_freq_port_list,
                                                              medium_freq_port_list, low_freq_port_list)
    port_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    training_ports_transform = np.array(training_ports_transform).reshape(len(training_ports_transform), 1)
    training_ports_encoded = port_encoder.fit_transform(training_ports_transform)
    print(f'training_ports_encoded: {training_ports_encoded[0:10, :]}')
    print(f'training_ports_encoded shape: {training_ports_encoded.shape}')

    ''' One hot encoding the protocol feature, it would produce a 3-dimensional vector'''
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')  # ignore will encode unseen protocol as all 0
    training_protocols = X_train[:, 1].reshape(len(X_train), 1)  # when keeping Dst port, protocol is the 2nd feature
    print(f'training_protocols unique: {np.unique(training_protocols)}')
    training_protocols_encoded = encoder.fit_transform(training_protocols)

    ''' normalize other features in the training set '''
    scaler = MinMaxScaler()
    X_train_scale = scaler.fit_transform(X_train[:, 2:])  # MinMax the rest of the features
    print(f'training_protocols_encoded: {training_protocols_encoded.shape}')
    print(f'X_train_scale: {X_train_scale.shape}')
    X_old = np.concatenate((training_ports_encoded, training_protocols_encoded), axis=1)
    X_old = np.concatenate((X_old, X_train_scale), axis=1)
    y_old = y_train.astype('int32')

    ''' normalize Test '''
    testing_ports = X_test[:, 0]
    testing_ports_transform = transform_ports_to_categorical(testing_ports, high_freq_port_list,
                                                              medium_freq_port_list, low_freq_port_list)
    testing_ports_transform = np.array(testing_ports_transform).reshape(len(testing_ports_transform), 1)
    testing_ports_encoded = port_encoder.transform(testing_ports_transform)
    print(f'testing_ports_encoded: {testing_ports_encoded[0:10, :]}')

    test_protocols = X_test[:, 1].reshape(len(X_test), 1)
    test_protocols_encoded = encoder.transform(test_protocols)

    X_test_scale = scaler.transform(X_test[:, 2:])
    X_new_normalize = np.concatenate((testing_ports_encoded, test_protocols_encoded), axis=1)
    X_new_normalize = np.concatenate((X_new_normalize, X_test_scale), axis=1)
    X_new = X_new_normalize
    y_new = y_test.astype('int32')

    print(f'X_old: {X_old.shape}, y_old: {y_old.shape}')
    print(f'X_new: {X_new.shape}, y_new: {y_new.shape}')
    print(f'y_old labels: {Counter(y_old)}')
    print(f'y_new labels: {Counter(y_new)}')
    # np.savez_compressed(save_path,
    #                     X_train=X_old, y_train=y_old,
    #                     X_test=X_new, y_test=y_new)
    # print('generated data file saved')

    stats(X_old, X_new, y_old, y_new)

    # take a look at the normalized X_new feature value (without readjusting max to 1)
    stats_data_helper(X_new_normalize, 'without adjusting max to 1')

    X_train = X_old, y_train = y_old, X_test=X_new, y_test=y_new
    # 筛掉类别
    data_filter_train = np.where(y_train != 5)[0]
    X_train = X_train[data_filter_train]
    y_train = y_train[data_filter_train]

    data_filter_train = np.where(y_train != 7)[0]
    X_train = X_train[data_filter_train]
    y_train = y_train[data_filter_train]

    data_filter_test = np.where(y_test != 7)[0]
    X_test = X_test[data_filter_test]
    y_test = y_test[data_filter_test]

    data_filter_test = np.where(y_test != 5)[0]
    X_test = X_test[data_filter_test]
    y_test = y_test[data_filter_test]

    # 修正标签为0-5
    print(f'before y_train Counter: {Counter(y_train)}')
    print(f'before y_test Counter: {Counter(y_test)}')

    family_idx = np.where(y_train == 6)[0]
    y_train[family_idx] = 5

    family_idx = np.where(y_test == 6)[0]
    y_test[family_idx] = 5

    # 保存模型
    print(f'before X_train: {X_train.shape}, y_train: {y_train.shape}')
    print(f'before X_test: {X_test.shape}, y_test: {y_test.shape}')
    print(f'before y_train Counter: {Counter(y_train)}')
    print(f'before y_test Counter: {Counter(y_test)}')

    np.savez_compressed(save_path,
                        X_train=X_train, y_train=y_train,
                        X_test=X_test, y_test=y_test)


def downsampling(X_train, y_train, ratio, phase):
    # Random sampling data
    '''
    Note here for benign traffic, we random sampling from all the benign traffic
    with no consideration of date. (since we only use one day of benign data)
    '''
    for idx, family in enumerate(np.unique(y_train)):
        family_idx = np.where(y_train == family)[0]
        family_size = len(family_idx)
        filter_idx = np.random.choice(family_idx, size=int(ratio * family_size), replace=False)
        X_train_family = X_train[filter_idx, :]
        y_train_family = y_train[filter_idx]
        print(f'idx: {idx}\tfamily: {family}')
        print(f'X_train_family: {X_train_family.shape}')
        print(f'y_train_family: {Counter(y_train_family)}\n\n')
        if idx == 0:
            X_train_sampling = X_train_family
            y_train_sampling = y_train_family
        else:
            X_train_sampling = np.concatenate((X_train_sampling, X_train_family), axis=0)
            y_train_sampling = np.concatenate((y_train_sampling, y_train_family), axis=0)
    return X_train_sampling, y_train_sampling

def transform_ports_to_categorical(ports, high_freq_port_list, medium_freq_port_list, low_freq_port_list):
    ports_transform = []
    for port in ports:
        if port in high_freq_port_list:
            ports_transform.append(0)
        elif port in medium_freq_port_list:
            ports_transform.append(1)
        else:
            ports_transform.append(2)
    return ports_transform


def stats_data_helper(X, data_type):
    print('==================')
    print(f'feature stats for {data_type}')
    print(f'min: {np.min(X, axis=0)}')
    print(f'avg: {np.average(X, axis=0)}')
    print(f'max: {np.max(X, axis=0)}')


def random_getmete(noisy_SAVE_FOLDER, save_name, ratio):
    data_file_path = os.path.join(noisy_SAVE_FOLDER, 'noise_data.npz')
    raw_data = np.load(data_file_path)
    X_train, y_train = raw_data['X_train'], raw_data['y_train']

    indx = np.arange(len(y_train))
    filter_idx = np.random.choice(indx, size=int(ratio * len(y_train)), replace=False)
    X_train_sampling = X_train[filter_idx, :]
    y_train_sampling = y_train[filter_idx]
    print(f'X_train_family: {X_train_sampling.shape}')
    print(f'y_train_family: {Counter(y_train_sampling)}\n\n')

    X_meta, y_meta = X_train_sampling, y_train_sampling
    print(f'X_meta: {X_meta.shape}, y_meta: {y_meta.shape}')
    print(f'y_meta labels: {Counter(y_meta)}')
    np.savez_compressed(os.path.join(noisy_SAVE_FOLDER, save_name),
                        X_meta=X_meta, y_meta=y_meta)
    print('generated data file saved')


def balance_getmete(noisy_SAVE_FOLDER, save_name, size):
    data_file_path = os.path.join(noisy_SAVE_FOLDER, 'noise_data.npz')
    raw_data = np.load(data_file_path)
    X_train, y_train = raw_data['X_train'], raw_data['y_train']

    family_sample_num = size

    for idx, family in enumerate(np.unique(y_train)):
        family_idx = np.where(y_train == family)[0]
        filter_idx = np.random.choice(family_idx, size=family_sample_num, replace=False)
        X_train_family = X_train[filter_idx, :]
        y_train_family = y_train[filter_idx]
        print(f'idx: {idx}\tfamily: {family}')
        print(f'X_train_family: {X_train_family.shape}')
        print(f'y_train_family: {Counter(y_train_family)}\n\n')
        if idx == 0:
            X_train_sampling = X_train_family
            y_train_sampling = y_train_family
        else:
            X_train_sampling = np.concatenate((X_train_sampling, X_train_family), axis=0)
            y_train_sampling = np.concatenate((y_train_sampling, y_train_family), axis=0)

    X_meta, y_meta = X_train_sampling, y_train_sampling

    print(f'X_meta: {X_meta.shape}, y_meta: {y_meta.shape}')
    print(f'y_meta labels: {Counter(y_meta)}')
    np.savez_compressed(os.path.join(noisy_SAVE_FOLDER, save_name),
                        X_meta=X_meta, y_meta=y_meta)
    print('generated data file saved')

def set_noise_label_asy(noise_ratio, NOISEDATA_SAVE_FOLDER, save_name):
    # 读取数据
    saved_noise_path = os.path.join(NOISEDATA_SAVE_FOLDER, 'noise_data.npz')
    raw_data = np.load(saved_noise_path)
    y_train = raw_data['y_train']
    print(f'before y_train Counter: {Counter(y_train)}')

    # asy标签
    y_noise = [y_train]
    for ratio in noise_ratio:
        y_train_ori = y_train.copy()
        for idx, family in enumerate(np.unique(y_train)):
            if family == 0:
                continue
            family_idx = np.where(y_train_ori == family)[0]
            family_size = len(family_idx)
            filter_idx = np.random.choice(family_idx, size=int(ratio * family_size), replace=False)
            y_train_ori[filter_idx] = 0
            print(f'ratio: {ratio}\tfamily: {family}\tstart: {family_size}\tpot: {len(filter_idx)}\tend: {family_size-len(filter_idx)}')
        print(f'y_train_family: {Counter(y_train_ori)}\n\n')
        y_noise.append(y_train_ori)
    saved_noise_label_path = os.path.join(NOISEDATA_SAVE_FOLDER, save_name)
    np.savez_compressed(saved_noise_label_path, y=y_noise[0], y1=y_noise[1], y3=y_noise[2],y5=y_noise[3],y7=y_noise[4])



def set_noise_label_sy(noise_ratio, NOISEDATA_SAVE_FOLDER, save_name):
    # 读取数据
    saved_noise_path = os.path.join(NOISEDATA_SAVE_FOLDER, 'noise_data.npz')
    raw_data = np.load(saved_noise_path)
    y_train = raw_data['y_train']
    print(f'before y_train Counter: {Counter(y_train)}')

    # sy标签
    y_noise = [y_train]
    for ratio in noise_ratio:
        y_train_ori = y_train.copy()
        y_train_noise = y_train.copy()
        for idx, family in enumerate(np.unique(y_train_ori)):
            if family == 0:
                family_idx = np.where(y_train_ori == family)[0]
                family_size = len(family_idx)
                filter_idx = np.random.choice(family_idx, size=int(ratio * family_size), replace=False)
                np.random.shuffle(filter_idx)
                filter_idx_list = np.array_split(filter_idx, 5)
                i = 0
                for idx1, family1 in enumerate(np.unique(y_train_ori)):
                    if family1 == 0 :
                        continue
                    y_train_noise[filter_idx_list[i]] = family1
                    i += 1
                print(
                    f'ratio: {ratio}\tfamily: {family}\tstart: {family_size}\tend: {family_size - len(filter_idx)}')
            else:
                family_idx = np.where(y_train_ori == family)[0]
                family_size = len(family_idx)
                filter_idx = np.random.choice(family_idx, size=int(ratio * family_size), replace=False)
                y_train_noise[filter_idx] = 0
                print(f'ratio: {ratio}\tfamily: {family}\tstart: {family_size}\tend: {family_size-len(filter_idx)+83145*0.2*ratio}')
        print(f'y_train_family: {Counter(y_train_noise)}\n\n')
        y_noise.append(y_train_noise)
    saved_noise_label_path = os.path.join(NOISEDATA_SAVE_FOLDER, save_name)
    np.savez_compressed(saved_noise_label_path, y=y_noise[0], y1=y_noise[1], y3=y_noise[2],y5=y_noise[3],y7=y_noise[4])



def set_noise_label_ask(NOISEDATA_SAVE_FOLDER, save_name):
    # 读取数据
    saved_noise_path = os.path.join(NOISEDATA_SAVE_FOLDER, 'noise_data.npz')
    raw_data = np.load(saved_noise_path)
    y_train = raw_data['y_train']
    print(f'before y_train Counter: {Counter(y_train)}')

    # ask标签
    y_noise_dict = {}
    ask_list = [3, 1]
    for asklabel in ask_list:
        y_train_ori = y_train.copy()
        y_train_noise = y_train.copy()

        orig_dist = np.bincount(y_train_ori, minlength=6)

        asklabel = int(asklabel)
        target_indices = np.where(y_train == asklabel)[0]
        # 如果没有目标标签，直接返回原数组
        if len(target_indices) == 0:
            print(f"警告: 目标标签 {asklabel} 不在数组中")
        # 生成随机数，为每个目标标签位置选择新标签
        # 其他标签列表（不包含目标标签）
        other_labels = [i for i in np.unique(y_train) if i != asklabel]
        # 为每个目标标签位置生成均匀分布的随机选择
        random_choices = np.random.choice(other_labels, size=len(target_indices))
        # 将目标标签位置的值替换为随机选择的新标签
        y_train_noise[target_indices] = random_choices

        # 计算最终标签分布
        final_dist = np.bincount(y_train_noise, minlength=6)

        print(f"原始标签分布: {orig_dist}")
        print(f"处理后标签分布: {final_dist}")
        print(f"修改后各标签的增量: {final_dist - orig_dist}")

        y_noise_dict[asklabel] = y_train_noise

    saved_noise_label_path = os.path.join(NOISEDATA_SAVE_FOLDER, save_name)
    np.savez_compressed(saved_noise_label_path, y=y_train, y3=y_noise_dict[3], y1=y_noise_dict[1])



if __name__ == '__main__':
    main()
