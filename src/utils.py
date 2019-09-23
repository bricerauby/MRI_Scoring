import pydicom
import os
import numpy as np
from natsort import natsorted
import tensorflow as tf
import SimpleITK as sitk

def load_dicom_dicom(path_to_folder, serie_name, type_strict=None):
    """loads a dicom and returns it as a np.array
     args :
     serie_name -- string, the name of the serie to load, it corresponds to the
     name of the folder containing all the dicom files
     path_to_folder -- string, the path to the folder containing all the series
     type_strict -- string, the type of MRI if None all type will be loaded
     otherwise only the matching exams
    """
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
    for filenameDCM in dcm_list:
        ds = pydicom.read_file(filenameDCM)
        assert ds.SeriesDescription == serie_description
        ArrayDicom[:, :, dcm_list.index(filenameDCM)] = ds.pixel_array
    return(infos, ArrayDicom)


def load_dicom(path_to_folder, serie_name, type_strict=None):
    """loads a dicom and returns it as a np.array
     args :
     serie_name -- string, the name of the serie to load, it corresponds to the
     name of the folder containing all the dicom files
     path_to_folder -- string, the path to the folder containing all the series
     type_strict -- string, the type of MRI if None all type will be loaded
     otherwise only the matching exams
    """
    serie_path = os.path.join(path_to_folder, serie_name)
    infos = {'serie_description': serie_name}
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(serie_path)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    const_voxel_dims = image.GetSize()
    infos['dims'] = const_voxel_dims
    ConstPixelSpacing = image.GetSpacing()
    infos['resolution'] = ConstPixelSpacing
    ArrayDicom = sitk.GetArrayFromImage(image)
    return(infos, ArrayDicom)


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def createExample(patient_infos, serie_name, path_to_folder):
    """Returns a tf.example create from a dicom serie_name
    arguments:
    serie_name -- string, the name of the serie to load, it corresponds to the
    name of the folder containing all the dicom files
    path_to_folder -- string, the path to the folder containing all the series
    patient_infos -- dict, the dictionary containing the infos about the
    patient of the serie
    """
    features = {
        'age': _int64_feature(patient_infos['age']),
        'Sequence_id': _bytes_feature(str(patient_infos['Sequence_id'])
                                      .encode('utf-8')),
        'EDSS': _float_feature(float(patient_infos['EDSS'])),
        'examination_date': _bytes_feature(patient_infos['examination_date']
                                           .encode('utf-8')),
    }

    exam_infos, ArrayDicom = load_dicom(path_to_folder, serie_name)
    stringDicom = ArrayDicom.tostring()
    serializedDicom = _bytes_feature(stringDicom)
    features['serie_description'] = (_bytes_feature
                                     (exam_infos['serie_description']
                                      .encode('utf-8')))
    h, w, d = exam_infos['dims']
    features['h'], features['w'], features['d'] = (_int64_feature(h),
                                                   _int64_feature(w),
                                                   _int64_feature(d))
    r_h, r_w, r_d = exam_infos['resolution']
    features['r_h'], features['r_w'], features['r_d'] = (_float_feature(r_h),
                                                         _float_feature(r_w),
                                                         _float_feature(r_d))
    features['image_raw'] = serializedDicom
    return(tf.train.Example(features=tf.train.Features(feature=features)))
