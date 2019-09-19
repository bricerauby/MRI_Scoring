
import os
import pprint
import tensorflow as tf

if 'COLAB_TPU_ADDR' not in os.environ:
    print('ERROR: Not connected to a TPU runtime; please see the first cell in'
          'this notebook for instructions!')
else:
    tpu_address = 'grpc://' + os.environ['COLAB_TPU_ADDR']
    print('TPU address is', tpu_address)

    with tf.Session(tpu_address) as session:
        devices = session.list_devices()

    print('TPU devices:')
    pprint.pprint(devices)
