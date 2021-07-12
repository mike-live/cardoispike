import numpy as np
import pandas as pd

FEATURES = [
            'x',
            'x_diff_1', 'x_diff_2','x_diff_3','x_diff_4',#'x_diff_5','x_diff_6',#'time_diff',
            'norm_diff_1', 'norm_diff_2','norm_diff_3','norm_diff_4',
            'mean_2','mean_4','mean_6',# 'mean_20', 'mean_50',
            'std_2','std_4','std_6', #'std_20', 'std_50',
            'norm_2','norm_4','norm_6', #'norm_20', 'norm_50',
            'diff_with_mean_2','diff_with_mean_4','diff_with_mean_6',
            'add_std_2', 'minus_std_2', 'add_2std_2', 'minus_2std_2', 'add_15std_2', 'minus_15std_2',
            'add_std_4', 'minus_std_4', 'add_2std_4', 'minus_2std_4', 'add_15std_4', 'minus_15std_4',
            'add_std_6', 'minus_std_6', 'add_2std_6', 'minus_2std_6', 'add_15std_6', 'minus_15std_6',
            'x_log_relative', 'rolling_mean', 'rolling_mean_rel'
]


def preprocess_df(df,
                  _id,
                  chunk_len=32,
                  is_one_hot_y=False,
                  x_column='x',
                  y_column='y',
                  N_CLASS=2):
    X = []
    Y = []
    # id2process_dct = {}

    seq_df = compute_seq_features(df, _id, chunk_len=chunk_len)
    # id2process_dct[_id] = seq_df

    # seq_df = id2process_dct[_id]
    seq_df = seq_df.fillna(0)
    seq_len = len(seq_df)

    for i in range(len(seq_df) - chunk_len):
        slice_df = seq_df.iloc[i:i+chunk_len]
        X.append(slice_df[FEATURES].values)
        y = slice_df['y'].tolist()[len(slice_df) // 2]
        if is_one_hot_y:
            # y = tf.keras.utils.to_categorical(y, num_classes=N_CLASS, dtype='float32')
            y = np.eye(N_CLASS, dtype='int')[y]

        Y.append(y)

    X = np.array(X, dtype='float32')
    Y = np.array(Y, dtype='int')

    return seq_df.time.values, X, seq_df.x.values, seq_df.y.values


def compute_seq_features(df, _id, chunk_len=32, aug=False):

    seq_df = df[df.id==_id].reset_index(drop=True)
    if aug:
        seq_df.x = add_noize(seq_df.x.values)

    x1 = np.mean(seq_df.x.values[:20])
    x2 = np.mean(seq_df.x.values[-20:])
    t0 = seq_df.time.values[0]
    t1 = seq_df.time.values[-1]

    start_df = []
    for i in range(chunk_len // 2):
        start_df.insert(0, [_id, t0 - (i + 1) * 600, x1, 0])

    end_df = []
    for i in range(chunk_len // 2):
        end_df.append([_id, t1 + (i + 1) * 600, x2, 0])

    start_df = pd.DataFrame(start_df, columns=['id', 'time', 'x', 'y'])
    end_df = pd.DataFrame(end_df, columns=['id', 'time', 'x', 'y'])
    seq_df = pd.concat([start_df, seq_df, end_df])

    seq_df['x_relative'] = seq_df.x / seq_df.x.shift(1)
    seq_df['x_log_relative'] = np.log(seq_df['x_relative'])
    seq_df = seq_df.fillna(method='ffill')

    seq_df['rolling_mean'] = seq_df['x'].rolling(window=5).max()
    seq_df['rolling_mean_rel'] = seq_df['x_log_relative'].rolling(window=5).max()

    seq_df['time_diff'] = seq_df.time.diff()
    for i in range(12):
        seq_df[f'x_diff_{i + 1}'] = seq_df.x.diff(i + 1).fillna(0)
    for i in range(12):
        seq_df[f'x_diff_front_{i + 1}'] = seq_df.x.diff(-(i + 1)).fillna(0)

    #################################### скользящие средние и дисперсии ###########################
    sizes = [2, 4, 6, 20, 50]
    for i in sizes:
        m, s = sliding(seq_df.x.values, i)
        seq_df[f'mean_{i}'] = m
        seq_df[f'std_{i}'] = s
        seq_df[f'add_std_{i}'] = (np.array(m) + np.array(s)) - np.array(seq_df.x.values)
        seq_df[f'minus_std_{i}'] = np.array(seq_df.x.values) - (np.array(m) - np.array(s))
        seq_df[f'add_2std_{i}'] = (np.array(m) + np.array(s) / 2) - np.array(seq_df.x.values)
        seq_df[f'minus_2std_{i}'] = np.array(seq_df.x.values) - (np.array(m) - np.array(s) / 2)
        seq_df[f'add_15std_{i}'] = (np.array(m) + 1.5 * np.array(s)) - np.array(seq_df.x.values)
        seq_df[f'minus_15std_{i}'] = np.array(seq_df.x.values) - (np.array(m) - 1.5 * np.array(s))
        seq_df[f'norm_{i}'] = (seq_df.x.values - np.array(m)) / (np.array(s) + 1e-3)
        seq_df[f'diff_with_mean_{i}'] = seq_df.x.values - np.array(m)

    for i in range(12):
        seq_df[f'norm_diff_{i + 1}'] = seq_df['norm_6'].diff(i + 1).fillna(0)

    for i in range(12):
        seq_df[f'norm_diff_front_{i + 1}'] = seq_df['norm_6'].diff(-(i + 1)).fillna(0)

    return seq_df


def sliding(x, len_):

    x = [x[0]] * (len_ // 2) + list(x) + [ x[-1] ] * (len_ // 2)
    mean, std = [], []
    for i in range(0, len(x) - len_, 1):
        mean.append(np.mean(x[i : i + len_]))
        std.append(np.std(x[i : i + len_]))

    return mean, std

def add_noize(a):
    return a + np.random.normal(0, 10, len(a))
