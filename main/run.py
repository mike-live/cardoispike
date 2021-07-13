import os
import sys
import pickle
import argparse

import numpy
import numpy as np
import pandas as pd
from tqdm import tqdm
from catboost import CatBoostClassifier

from preprocess import preprocess_df


default_weights_path = 'weights/12.07.21/cardiospike.cbm'
default_pca_path = 'weights/12.07.21/pca.pkl'
CHUNK_LEN = 10


def load_all(data_path, model_path, pca_path, verbose=True):
    # load data
    data = pd.read_csv(data_path) #args.data_path)
    if verbose:
        print('Data loaded!')

    # load model
    model = CatBoostClassifier()
    model.load_model(model_path) #args.weights_path)
    if verbose:
        print('CatBoost model loaded!')

    # load pca coefficients
    with open(pca_path, 'rb') as pickle_file:
        pca_transformer = pickle.load(pickle_file)
    if verbose:
        print('PCA coefficients loaded!')

    return data, model, pca_transformer


def inference(df, model, pca_transformer, target_column='y') -> numpy.ndarray:

    for _id in tqdm(df.id.unique()):
        df[target_column] = 0
        time, x_data, x, y = preprocess_df(df, _id, chunk_len=CHUNK_LEN)
        pca = pca_transformer.transform(x_data[:,:, 0])
        x_data = x_data.reshape((x_data.shape[0], x_data.shape[1] * x_data.shape[2] ))

        x_data = x_data.tolist()
        for i in range(len(pca)):
            x_data[i].extend(pca[i])
        x_data = np.array(x_data)

        pred = model.predict(x_data)
        df[target_column][df.id == _id] = pred
        pred = np.array(pred)

    return df, pred

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', required=True, type=str, help='Path to R-R intervals data .csv file')
    parser.add_argument('--output_path', required=False, default='../', type=str, help='Path to output .csv file')
    parser.add_argument('--output_filename', required=False, default='output.csv', type=str, help='Path to output .csv file')
    parser.add_argument('--weights_path', required=False, type=str, default=default_weights_path, help='CatBoost model weights path')
    parser.add_argument('--pca_path', required=False, type=str, default=default_pca_path, help='PCA saved coefficients path')
    parser.add_argument('--verbose', required=False, type=bool, default=True, help='Verbosity switcher')
    args = parser.parse_args()

    try:
        data, model, pca_transformer = load_all(data_path=args.data_path,
                                                model_path=args.weights_path,
                                                pca_path=args.pca_path,
                                                verbose=args.verbose)
    except Exception as error:
        print(error)
        print('Loading failed!')
        sys.exit(0)

    output_df, pred = inference(df=data,
                                model=model,
                                pca_transformer=pca_transformer)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    output_filename = os.path.join(args.output_path, args.output_filename)
    output_df.to_csv(output_filename)

    if args.verbose:
        print(f'Prediction saved to {output_filename}')
