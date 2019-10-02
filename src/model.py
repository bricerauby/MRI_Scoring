
from tensorflow import keras
from tensorflow.keras import layers


def dumb_model():
    inputs = layers.Input(shape=(None, None, None, 1))
    x = inputs
    x = layers.Conv3D(1,
                      kernel_size=3,
                      strides=1,
                      padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=keras.regularizers.l2(1e-4),
                      use_bias=True
                      )(x)
    return(keras.Model(inputs=inputs, outputs=x))


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 use3D=False):
    """3D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv3D number of filters
        kernel_size (int): Conv3D square kernel dimensions
        strides (int): Conv3D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    if use3D:
        conv = layers.Conv3D(num_filters,
                             kernel_size=kernel_size,
                             strides=strides,
                             padding='same',
                             kernel_initializer='he_normal',
                             kernel_regularizer=keras.regularizers.l2(1e-4),
                             use_bias=(not batch_normalization)
                             )
    else:
        conv = layers.Conv2D(num_filters,
                             kernel_size=kernel_size,
                             strides=strides,
                             padding='same',
                             kernel_initializer='he_normal',
                             kernel_regularizer=keras.regularizers.l2(1e-4),
                             use_bias=(not batch_normalization)
                             )
    x = inputs
    x = conv(x)
    if batch_normalization:
        x = layers.BatchNormalization()(x)
    if activation is not None:
        x = layers.Activation(activation)(x)
    return x

def conv(num_filters=64, kernel_size=3, strides=1, 
         padding='same', kernel_initializer='he_normal', 
         kernel_regularizer=keras.regularizers.l2(1e-4), 
         useBias=False,
         use3D=True): 
    if use3D:
        return layers.Conv3D(num_filters, 
                             kernel_size=kernel_size, 
                             strides=strides, 
                             padding='same', 
                             kernel_initializer='he_normal', 
                             kernel_regularizer=keras.regularizers.l2(1e-4), 
                             use_bias=useBias)
    else:
        return layers.Conv2D(num_filters, 
                             kernel_size=kernel_size, 
                             strides=strides, 
                             padding='same', 
                             kernel_initializer='he_normal', 
                             kernel_regularizer=keras.regularizers.l2(1e-4), 
                             use_bias=useBias)      
           
           
def conv_unit(inputs, num_filters=64,
     kernel_size=3,
     activation='relu',
     batch_normalization=True,
     use3D=False): 
    
    if use3D:
        max_pool = layers.MaxPooling3D(pool_size=3, strides=2, padding='same', data_format=None)
    else:
        max_pool = layers.MaxPooling2D(pool_size=3, strides=2, padding='same', data_format=None)
        
    x = inputs 
    path1, path2, path3, path4 = (conv(strides=1, use3D=use3D, useBias=(not batch_normalization), num_filters=num_filters)(x),  
                                  conv(strides=1, use3D=use3D, useBias=(not batch_normalization), num_filters=num_filters)(x),  
                                  conv(strides=2, use3D=use3D, useBias=(not batch_normalization), num_filters=num_filters)(x), 
                                  max_pool(x))
    if batch_normalization:
        path1, path2, path3 = (layers.BatchNormalization()(path1), 
                               layers.BatchNormalization()(path2), 
                               layers.BatchNormalization()(path3))
    if activation is not None:
        path1, path2, path3 = (layers.Activation(activation)(path1), 
                               layers.Activation(activation)(path2), 
                               layers.Activation(activation)(path3))
    path1, path2 =  (conv(strides=1, use3D=use3D, useBias=(not batch_normalization), num_filters=num_filters)(path1),
                     conv(strides=2, use3D=use3D, useBias=(not batch_normalization), num_filters=num_filters)(path2))
    if batch_normalization:
        path1, path2 = (layers.BatchNormalization()(path1), 
                        layers.BatchNormalization()(path2))
                              
    if activation is not None:
        path1, path2 = (layers.Activation(activation)(path1), 
                        layers.Activation(activation)(path2)) 

    path1 = conv(strides=2, use3D=use3D, useBias=(not batch_normalization), num_filters=num_filters)(path1) 
    if batch_normalization:
        path1 = layers.BatchNormalization()(path1)
                              
    if activation is not None:
        path1 = layers.Activation(activation)(path1)
    x = layers.Concatenate()([path1, path2, path3, path4])
    return x
    

def resnet(input_shape, depth, num_classes, use3D=False, useBatchNorm=True):
    """ResNet Version 1 Model builder [a]

    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is downsampled
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = layers.Input(shape=input_shape)
    x = resnet_layer(inputs=inputs, use3D=use3D)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides,
                             batch_normalization=useBatchNorm,
                             use3D=use3D)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None,
                             batch_normalization=useBatchNorm,
                             use3D=use3D)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False,
                                 use3D=use3D)
            x = layers.add([x, y])
            x = layers.Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    if use3D:
        x = layers.AveragePooling3D(pool_size=8)(x)
    else:
        x = layers.AveragePooling2D(pool_size=8)(x)
    y = layers.Flatten()(x)
    outputs = layers.Dense(num_classes,
                           activation='softmax',
                           kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def customInception(input_shape,  num_classes, num_filters=64, dense_size=256, use3D=False, useBatchNorm=True):
    inputs = layers.Input(shape=input_shape)
    x = inputs 
    for i in range(3):
        x = conv_unit(x, use3D=use3D, batch_normalization=useBatchNorm, num_filters=num_filters)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    y = layers.Dense(dense_size, activation='softmax', kernel_initializer='he_normal')(x)
    y = layers.Dropout(0.5)(y)
    outputs = layers.Dense(num_classes,
                           activation='softmax',
                           kernel_initializer='he_normal')(y)
    # Instantiate model.
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model    