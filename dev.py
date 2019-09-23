import os
from src.utils import load_dicom as loader
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import pydicom
import os
import numpy as np
from natsort import natsorted
import tensorflow as tf

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


path_to_folder = 'data/Dataset_1'
serie_name = '55457'



# labels = pd.read_csv(os.path.join(path_to_folder, 'labels.csv'), decimal=",")
# patient_infos = labels[labels.Sequence_id == int(serie_name)]
# features = {
#     'age': _int64_feature(patient_infos['age'].iloc[0]),
#     'Sequence_id': _bytes_feature(str(patient_infos['Sequence_id'].iloc[0])
#                                   .encode('utf-8')),
#     'EDSS': _float_feature(float(patient_infos['EDSS'].iloc[0])),
#     'examination_date': _bytes_feature(patient_infos['examination_date'].iloc[0]
#                                        .encode('utf-8')),
# }
#
# exam_infos, ArrayDicom = loader(path_to_folder, serie_name)
# stringDicom = ArrayDicom.tostring()
# serializedDicom = _bytes_feature(stringDicom)
# features['serie_description'] = _bytes_feature(exam_infos['serie_description']
#                                                .encode('utf-8'))
# h, w, d = exam_infos['dims']
# features['h'], features['w'], features['d'] = (_int64_feature(h),
#                                                _int64_feature(w),
#                                                _int64_feature(d))
# r_h, r_w, r_d = exam_infos['resolution']
# features['r_h'], features['r_w'], features['r_d'] = (_float_feature(r_h),
#                                                      _float_feature(r_w),
#                                                      _float_feature(r_d))
# features['image_raw'] = serializedDicom
# example = tf.train.Example(features=tf.train.Features(feature=features))


serie_path = os.path.join(path_to_folder, serie_name)
infos = {'id': serie_name}
dcm_list = []
for filename in os.listdir(serie_path):
    if filename.endswith('.dcm'):
        dcm_list.append(os.path.join(serie_path, filename))
dcm_list = natsorted(dcm_list)
RefDs = pydicom.read_file(dcm_list[0])
serie_description = RefDs.SeriesDescription
infos['serie_description'] = serie_description
const_voxel_dims = (int(RefDs.Rows), int(RefDs.Columns), len(dcm_list))
infos['dims'] = const_voxel_dims
ConstPixelSpacing = (float(RefDs.PixelSpacing[0]),
                     float(RefDs.PixelSpacing[1]),
                     float(RefDs.SliceThickness))
infos['resolution'] = ConstPixelSpacing
ArrayDicom = np.zeros(const_voxel_dims, dtype=RefDs.pixel_array.dtype)
print('here')
for filenameDCM in dcm_list:
    ds = pydicom.read_file(filenameDCM)
    assert ds.SeriesDescription == serie_description
    ArrayDicom[:, :, dcm_list.index(filenameDCM)] = ds.pixel_array

#
# record_iterator = tf.python_io.tf_record_iterator(path='images.tfrecords')
#
# for string_record in record_iterator:
#     example = tf.train.Example()
#     example.ParseFromString(string_record)
#     plt.imshow(example['image_raw'].numpy()[:, :, 50])
# raw_image_dataset = tf.data.TFRecordDataset('images.tfrecords')

# Create a dictionary describing the features.
image_feature_description = {
    'h': tf.FixedLenFeature([], tf.int64),
    'w': tf.FixedLenFeature([], tf.int64),
    'd': tf.FixedLenFeature([], tf.int64),
    'r_h': tf.FixedLenFeature([], tf.float32),
    'r_w': tf.FixedLenFeature([], tf.float32),
    'r_d': tf.FixedLenFeature([], tf.float32),
    'EDSS': tf.FixedLenFeature([], tf.float32),
    'age': tf.FixedLenFeature([], tf.int64),
    'Sequence_id': tf.FixedLenFeature([], tf.string),
    'examination_date': tf.FixedLenFeature([], tf.string),
    'image_raw': tf.FixedLenFeature([], tf.string),
}

raw_image_dataset = tf.data.TFRecordDataset('images.tfrecords')


def _parse_image_function(example_proto):
    # Parse the input tf.Example proto using the dictionary above.
    return tf.parse_single_example(example_proto, image_feature_description)


parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
print(parsed_image_dataset)


# serie_path = os.path.join(path_to_folder,serie_name)
# infos = {'name':serie_name}
# dcm_list = []
# for filename in os.listdir(serie_path):
#     if filename.endswith('.dcm'):
#         dcm_list.append(os.path.join(serie_path,filename))
# dcm_list = natsorted(dcm_list)
# RefDs = pydicom.read_file(dcm_list[0])
# serie_description = RefDs.SeriesDescription
# print(serie_description)
# print(RefDs.)
# print(RefDs.__dict__)

#
#
#
# for serie_name in os.listdir(path_to_folder):
#     try:
#         serie_path = os.path.join(path_to_folder,serie_name)
#         infos = {'name':serie_name}
#         dcm_list = []
#         for filename in os.listdir(serie_path):
#             if filename.endswith('.dcm'):
#                 dcm_list.append(os.path.join(serie_path,filename))
#         dcm_list = natsorted(dcm_list)
#         RefDs = pydicom.read_file(dcm_list[0])
#         serie_description = RefDs.SeriesDescription
#         print(serie_description)
#     except :
#         pass
