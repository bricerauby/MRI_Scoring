import tensorflow as tf
import argparse
import numpy as np
import os
import pandas as pd
import tqdm
from utils import createExample
import json

parser = argparse.ArgumentParser()
parser.add_argument("dataset_path", help="the path to the dataset, contains"
                                         "the labels.csv and the dicom folder",
                    type=str)
parser.add_argument("tf_record_path", help="the path to output the tf record",
                    type=str)
parser.add_argument("seed", help="the seed for the split.",
                    type=int, default=0)
parser.add_argument("test_ratio", help="the ratio size_test/size_dataset.",
                    type=float, default=0)


def generate_tfrecords(path_to_folder, tf_record_path,
                       labels, id_series, mode):
    print('Generating tfrecords for {} set ....'.format(mode))
    for serie in tqdm.tqdm(id_series):
        # print(serie)
        try:
            patient_infos = labels[labels.Sequence_id == int(serie)]
            dict_infos = {
                'age': patient_infos['age'].iloc[0],
                'Sequence_id': patient_infos['Sequence_id'].iloc[0],
                'EDSS': patient_infos['EDSS'].iloc[0],
                'examination_date': patient_infos['examination_date'].iloc[0]
            }
            example = createExample(dict_infos, str(serie), path_to_folder)
            with tf.io.TFRecordWriter(os.path.join(tf_record_path,
                                                   '{}_set_{}.tfrecords')
                                             .format(mode, serie)) as writer:
                writer.write(example.SerializeToString())
        except NotImplementedError:
            print('{} has been ignored'.format(serie))
            with open(os.path.join(tf_record_path,
                                   'ignored.txt'), 'a') as file:
                file.write('{} NotImplemented \n'.format(serie))
        except ValueError:
            print('{} has been ignored'.format(serie))
            with open(os.path.join(tf_record_path,
                                   'ignored.txt'), 'a') as file:
                file.write('{} ValueError \n'.format(serie))
        except AssertionError:
            print('{} has been ignored'.format(serie))
            with open(os.path.join(tf_record_path,
                                   'ignored.txt'), 'a') as file:
                file.write('{} AssertionError \n'.format(serie))
    print('tfrecords for {} set Generated !'.format(mode))


def main(raw_args=None):
    args = parser.parse_args(raw_args)
    path_to_folder = args.dataset_path
    tf_record_path = args.tf_record_path
    if not os.path.exists(tf_record_path):
        os.makedirs(tf_record_path)
    np.random.seed(args.seed)
    labels = pd.read_csv(os.path.join(path_to_folder, 'labels.csv'),
                         decimal=",")
    list_series = labels.Sequence_id.tolist()
    list_series = np.random.permutation(list_series)
    n_test_set = int(len(list_series) * args.test_ratio)
    test_series = list_series[:n_test_set]
    train_series = list_series[n_test_set:]
    split_dict = {'train_id': list(map(str, train_series)),
                  'test_id': list(map(str, test_series))}
    with open(os.path.join(tf_record_path, 'split.json'), 'w') as json_file:
        json.dump(split_dict, json_file)
    generate_tfrecords(path_to_folder, tf_record_path,
                       labels, train_series, 'train')
    generate_tfrecords(path_to_folder, tf_record_path,
                       labels, test_series, 'test')


if __name__ == '__main__':
    main()
