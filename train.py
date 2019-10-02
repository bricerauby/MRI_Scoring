import src.model
import numpy as np 
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import os
import json 
import datetime
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument("--path_to_data", type=str,
                help="path to the repo containing the tfrecords directory")

parser.add_argument("--log_dir", type=str,
                help="the directory in which the log from training will be stored")
parser.add_argument("--config_file", type=str,
                help="path to the config.json")

def parse_fn(example):
    "Parse TFExample records and perform simple data augmentation."
    example_fmt = {
        'age': tf.io.FixedLenFeature((), tf.int64, -1),
        'Sequence_id': tf.io.FixedLenFeature((), tf.string, ""),
        'EDSS': tf.io.FixedLenFeature((), tf.float32, 0.),
        'examination_date': tf.io.FixedLenFeature((), tf.string, ""),
        'serie_description': tf.io.FixedLenFeature((), tf.string, ""),
        'h': tf.io.FixedLenFeature((), tf.int64, 0),
        'w': tf.io.FixedLenFeature((), tf.int64, 0),
        'd': tf.io.FixedLenFeature((), tf.int64, 0),
        'r_h': tf.io.FixedLenFeature((), tf.float32, 0),
        'r_w': tf.io.FixedLenFeature((), tf.float32, 0),
        'r_d': tf.io.FixedLenFeature((), tf.float32, 0),
        'image_raw':  tf.io.FixedLenFeature((), tf.string, "")
    }
    parsed = tf.parse_single_example(example, example_fmt)
    image = tf.decode_raw(parsed['image_raw'],out_type=tf.int16)
    image = tf.reshape(image, shape=(parsed['h'],
                                     parsed['w'],
                                     parsed['d'],
                                     1) )

    return image, parsed["EDSS"]

def input_fn(path_to_tf, mode, batch_size=1, buffer_size=1, prefetch_buffer_size=1, num_workers=8):
    files = tf.data.Dataset.list_files(os.path.join(path_to_tf, '{}_set_*.tfrecords'.format(mode)))
    dataset = files.apply(tf.contrib.data.parallel_interleave(tf.data.TFRecordDataset, 
                                                              cycle_length=num_workers))
    dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.repeat()
    dataset = dataset.map(map_func=parse_fn, num_parallel_calls=num_workers)
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(buffer_size=prefetch_buffer_size)
    return dataset



def main(raw_args=None):
    args = parser.parse_args(raw_args)
    with open(args.config_file) as file: 
        hyper_params = json.load(file)
    path_to_tf = os.path.join(args.path_to_data, 
                              "tf_records_{}_seed_{}/".format(hyper_params['input_size'], 
                                                              hyper_params['seed']))
    log_dir = args.log_dir

    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    with open(os.path.join(path_to_tf, 'split.json')) as file: 
        split = json.load(file)


    time = datetime.datetime.today()
    log_id = '{}_{}h{}min'.format(time.date(), time.hour, time.minute)
    log_path = os.path.join(log_dir,log_id)
    i = 0
    while os.path.exists(log_path):
        i = 1
        log_id ='{}_{}h{}min_{}'.format(time.date(), time.hour, time.minute, i)
        log_path = os.path.join(log_dir,log_id)
    os.mkdir(log_path)

    with open(os.path.join(log_path,'split.json'), 'w') as fp:
        json.dump(split, fp)
    with open(os.path.join(log_path,'hyper_params.json'), 'w') as fp:
        json.dump(hyper_params, fp)


    n_train = len(split['train_id'])
    n_valid = len(split['valid_id']) 
    train_steps_per_epoch = n_train//hyper_params['batch_size']
    valid_steps_per_epoch = n_valid//hyper_params['batch_size']

    train_dataset = input_fn(path_to_tf,'train', batch_size=hyper_params['batch_size'])
    valid_dataset = input_fn(path_to_tf,'test', batch_size= hyper_params['batch_size'])
    if hyper_params['model'] == 'resnet':
        model = src.model.resnet((hyper_params['input_size'], 
                                  hyper_params['input_size'], 
                                  hyper_params['input_size'], 1), 
                                  32, 
                                  1, 
                                  use3D=True, 
                                  useBatchNorm=hyper_params['batch_norm'])
    if hyper_params['model'] == 'customInception':
        model = src.model.customInception((hyper_params['input_size'], 
                                  hyper_params['input_size'], 
                                  hyper_params['input_size'], 1), 
                                  1, 
                                  num_filters=hyper_params['num_filters'], 
                                  dense_size=hyper_params['dense_size'],
                                  use3D=True, 
                                  useBatchNorm=hyper_params['batch_norm'])
    adam = Adam(lr=hyper_params['learning_rate'])
    model.compile(optimizer=adam,
                  loss= 'mean_squared_error',
                  metrics=['mean_squared_error'])


    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                  patience=3, min_lr=hyper_params['minimum_learning_rate'], verbose=1)
    save_best = ModelCheckpoint(os.path.join(log_path,'weights.{epoch:02d}-{val_loss:.2f}.hdf5'), monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    call_backs = [reduce_lr, save_best]  
    model.fit(train_dataset, 
              epochs=hyper_params['n_epochs'], 
              steps_per_epoch=train_steps_per_epoch, 
              validation_data=valid_dataset,
              validation_steps=valid_steps_per_epoch,
              callbacks=call_backs)

                              
if __name__ == '__main__': 
    main()