
# coding: utf-8

''' SECTION 1: Some variables need to be specified
##
# '''

# This code is modifed from https://github.com/martinarjovsky/WassersteinGAN 
# get_ipython().magic(u'matplotlib inline')

import os
os.environ['KERAS_BACKEND']='tensorflow' # tensorflow
os.environ["CUDA_VISIBLE_DEVICES"]="0" # set the id number of GPU

## Specify where to save the trained model
save_file_name = 'depgan_twoCritics_prob_noSL_21102019'

## Specify the location of .txt files for accessing the training data
config_dir = 'train_data_server_fold'

# Threshold which defines what is WMH in both probability map (PM)
# and ireegularity map (IM). PM should use 0.500 while IM should
# use 0.178.
IM_TRSH = 0.500

''' SECTION 2: Call libraries
##
# '''

import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Conv2D, ZeroPadding2D, BatchNormalization, Input, Dropout, Reshape
from keras.layers import UpSampling2D, MaxPooling2D, AveragePooling2D, Dense, Lambda, Conv1D
from keras.layers import Conv2DTranspose, Reshape, Activation, Cropping2D, Flatten
from keras.layers.merge import concatenate, multiply, add
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import RMSprop, SGD, Adam
from keras.activations import relu
from keras.initializers import RandomNormal
conv_init = RandomNormal(0, 0.02)
gamma_init = RandomNormal(1., 0.02)

import h5py, os, math
import numpy as np
from sklearn.model_selection import train_test_split

import nibabel as nib
import numpy as np
import tensorflow as tf
import time

from StringIO import StringIO
import matplotlib.pyplot as plt

# To clear the defined variables and operations of the previous cell
tf.reset_default_graph() 

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

''' SECTION 3: Define classes and functions
##
# '''

# Class to load NifTi (.nii) data
class load_data(object):
    # Load NII data
    def __init__(self, image_name):
        # Read nifti image
        nim = nib.load(image_name)
        image = nim.get_data()
        affine = nim.affine
        self.image = image
        self.affine = affine
        self.dt = nim.header['pixdim'][4]

# Function to prepare the data after opening from .nii file
def data_prep(image_data):
    # Extract the 2D slices from the cardiac data
    image = image_data.image
    images = []
    for z in range(image.shape[2]):
        image_slice = image[:, :, z]
        images += [image_slice]
    images = np.array(images, dtype='float32')

    # Both theano and caffe requires input array of dimension (N, C, H, W)
    # TensorFlow (N, H, W, C)
    # Add the channel dimension, swap width and height dimension
    images = np.expand_dims(images, axis=3)

    return images

# Function to prepare the data before saving to .nii file
def data_prep_save(image_data):
    image_data = np.squeeze(image_data)
    output_img = np.swapaxes(image_data, 0, 2)
    output_img = np.rot90(output_img)
    output_img = output_img[::-1,...]   # flipped

    return output_img

# Change integer values of a matrix into real values (based on min and max values) 
def map_image_to_intensity_range(image, min_o, max_o, percentiles=0):

    # If percentile = 0 uses min and max. Percentile >0 makes normalisation more robust to outliers.

    if image.dtype in [np.uint8, np.uint16, np.uint32]:
        assert min_o >= 0, 'Input image type is uintXX but you selected a negative min_o: %f' % min_o

    if image.dtype == np.uint8:
        assert max_o <= 255, 'Input image type is uint8 but you selected a max_o > 255: %f' % max_o

    min_i = np.percentile(image, 0 + percentiles)
    max_i = np.percentile(image, 100 - percentiles)

    image = (np.divide((image - min_i), max_i - min_i) * (max_o - min_o) + min_o).copy()

    image[image > max_o] = max_o
    image[image < min_o] = min_o

    return image

# Calculate Dice coefficient score
# https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py#L19
def dice_coef(y_true, y_pred, smooth=1e-7):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# Calculate Dice coefficient loss
# https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py#L19
def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

### Simple example on how to log scalars and images to tensorboard without tensor ops.
## License: Copyleft
# __author__ = "Michael Gygli"
class Logger(object):
    """Logging in tensorboard without tensorflow ops."""

    def __init__(self, log_dir):
        """Creates a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(log_dir)
        
    def log_graph(self, sess):
        self.writer.add_graph(sess.graph)

    def log_scalar(self, tag, value, step):
        """Log a scalar variable.
        Parameter
        ----------
        tag : basestring
            Name of the scalar
        value
        step : int
            training iteration
        """
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag,
                                                     simple_value=value)])
        self.writer.add_summary(summary, step)

    def log_images(self, tag, images, step, dtype='RGB', denorm=[0,255]):
        """Logs a list of images."""

        im_summaries = []
        for nr, img in enumerate(images):
            # Write the image to a string
            s = StringIO()
            if dtype == 'RGB':
                img = ( (img+1)/2*denorm[1]).clip(denorm[0],denorm[1]).astype('uint8')
                plt.imsave(s, img, format='png')
            else:
                # img_t = np.concatenate((img, img), axis=-1)
                # img_t = np.concatenate((img_t, img), axis=-1)
                plt.imsave(s, np.squeeze(img), cmap='viridis', format='png')                

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            # Create a Summary value
            im_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, nr),
                                                 image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=im_summaries)
        self.writer.add_summary(summary, step)
        
    def log_histogram(self, tag, values, step=0, bins=1000):
        """Logs the histogram of a list/vector of values."""
        # Convert to a numpy array
        values = np.array(values)
        
        # Create histogram using numpy        
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill fields of histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
        # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
        # Thus, we drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()

''' SECTION 4: Build the networks
##
# '''

# Dense layer
def dense_bn(units=32, name_to_concat='', input_tensor=None):
    tensor = Dense(units, kernel_initializer='he_normal', name=str('dense_'+name_to_concat))(input_tensor)
    tensor = BatchNormalization(name=str('dense_bn_'+name_to_concat))(tensor)

    return tensor

# Dense layer with ReLU
def dense_bn_relu(units=32, padding='same', name_to_concat='', input_tensor=None):
    tensor = Dense(units, kernel_initializer='he_normal', name=str('dense_'+name_to_concat))(input_tensor)
    tensor = BatchNormalization(name=str('dense_bn_'+name_to_concat))(tensor)
    tensor = Activation('relu', name=str('dense_relu_'+name_to_concat))(tensor)

    return tensor

# 1D conv layer
def conv1d_bn(filter_size=3, filter_number=32, padding='same', name_to_concat='', input_tensor=None):
    tensor = Conv1D(filter_number, filter_size, padding=padding, name=str('conv1d_'+name_to_concat))(input_tensor)
    tensor = BatchNormalization(name=str('bn_'+name_to_concat))(tensor)

    return tensor

# 1D conv layer with ReLU
def conv1d_bn_relu(filter_size=3, filter_number=32, padding='same', name_to_concat='', input_tensor=None):
    tensor = Conv1D(filter_number, filter_size, padding=padding, name=str('conv1d_'+name_to_concat))(input_tensor)
    tensor = BatchNormalization(name=str('bn_'+name_to_concat))(tensor)
    tensor = Activation('relu', name=str('relu_'+name_to_concat))(tensor)

    return tensor

# 2D conv layer with ReLU
def conv2d_bn_relu(filter_size=3, filter_number=32, padding='same', name_to_concat='', input_tensor=None):
    tensor = Conv2D(filter_number, (filter_size, filter_size), padding=padding, name=str('conv2d_'+name_to_concat))(input_tensor)
    tensor = BatchNormalization(name=str('bn_'+name_to_concat))(tensor)
    tensor = Activation('relu', name=str('relu_'+name_to_concat))(tensor)

    return tensor

# 2D conv layer
def conv2d_bn(filter_size=3, filter_number=32, padding='same', name_to_concat='', input_tensor=None):
    tensor = Conv2D(filter_number, (filter_size, filter_size), padding=padding, name=str('conv2d_'+name_to_concat))(input_tensor)
    tensor = BatchNormalization(name=str('bn_'+name_to_concat))(tensor)

    return tensor

# 2D conv layer no BN
def conv2d_relu(filter_size=3, filter_number=16, padding='same', name_to_concat='', input_tensor=None):
    tensor = Conv2D(filter_number, (filter_size, filter_size), padding=padding, name=str('conv2d_'+name_to_concat))(input_tensor)
    tensor = Activation('relu', name=str('relu_'+name_to_concat))(tensor)

    return tensor

# 2D conv layer with ReLU no BN
def deconv2d_bn_relu(filter_size=4, filter_number=16, padding='valid', name_to_concat='', input_tensor=None):
    tensor = Conv2DTranspose(filter_number, (filter_size, filter_size), strides=(2, 2), padding=padding, name=str('deconv2d_'+name_to_concat))(input_tensor)
    tensor = BatchNormalization(name=str('bn_'+name_to_concat))(tensor)
    tensor = Activation('relu', name=str('relu_'+name_to_concat))(tensor)

    return tensor


## Critic network
def Dis_C2D_FCN1(input_shape):
    inputs = Input(input_shape, name='input_dis')

    conv_0 = conv2d_relu(filter_size=5, filter_number=16, name_to_concat='dis_0a', input_tensor=inputs)
    conv_0 = conv2d_relu(filter_size=5, filter_number=16, name_to_concat='dis_0b', input_tensor=conv_0)
    pool_0 = MaxPooling2D(pool_size=(2, 2), name='maxpool2d_dis_0')(conv_0)

    conv_1 = conv2d_relu(filter_size=5, filter_number=32, name_to_concat='dis_1a', input_tensor=pool_0)
    conv_1 = conv2d_relu(filter_size=5, filter_number=32, name_to_concat='dis_1b', input_tensor=conv_1)
    pool_1 = MaxPooling2D(pool_size=(2, 2), name='maxpool2d_dis_1')(conv_1)

    conv_2 = conv2d_relu(filter_size=3, filter_number=64, name_to_concat='dis_2', input_tensor=pool_1)
    conv_3 = conv2d_relu(filter_size=3, filter_number=64, name_to_concat='dis_3', input_tensor=conv_2)
    pool_2 = MaxPooling2D(pool_size=(2, 2), name='maxpool2d_dis_2')(conv_3)

    conv_4 = conv2d_relu(filter_size=3, filter_number=128, name_to_concat='dis_4', input_tensor=pool_2)
    conv_5 = conv2d_relu(filter_size=3, filter_number=128, name_to_concat='dis_5', input_tensor=conv_4)
    pool_3 = MaxPooling2D(pool_size=(2, 2), name='maxpool2d_dis_3')(conv_5)

    conv_6 = conv2d_relu(filter_size=3, filter_number=256, name_to_concat='dis_6', input_tensor=pool_3)
    conv_7 = conv2d_relu(filter_size=3, filter_number=256, name_to_concat='dis_7', input_tensor=conv_6)

    conv_8 = conv2d_relu(filter_size=3, filter_number=256, name_to_concat='dis_8', input_tensor=conv_7)
    conv_9 = Conv2D(1, (1, 1), kernel_initializer='he_normal', name='dis_9')(conv_8)

    flat_0 = Flatten()(conv_9)
    dense_1 = Dense(1, kernel_initializer='he_normal')(flat_0)

    model = Model(inputs=inputs, outputs=dense_1, name='Dis_C2D_FCN1')
    return model


## Generator network
def Gen_UNet2D(input_shape, noiseZ_shape=(32, 1), first_fm=32, nc_out=1):
    n_ch_0 = first_fm
    inputs = Input(input_shape, name='input_gen_chn_0')

    # Inputting the noises
    noiseZ_shape_batch = (noiseZ_shape[0], noiseZ_shape[1])
    inputNoiseZ = Input(noiseZ_shape_batch, name='input_gen_noiseZ_0')    

    # Noise layers - Firsts
    noise_1_add_f = dense_bn_relu(units=n_ch_0, name_to_concat='noise_1_add_f0', input_tensor=inputNoiseZ)
    noise_1_add_f = dense_bn_relu(units=n_ch_0, name_to_concat='noise_1_add_f1', input_tensor=noise_1_add_f)
    noise_1_add_f = Flatten()(noise_1_add_f)

    # Modulation of Noise layers - Addition+++
    noise_2_add_m3 = dense_bn(units=n_ch_0*3, name_to_concat='noise_2_add_m3', input_tensor=noise_1_add_f)
    # Modulation of Noise layers - Multiplication+++
    noise_2_mul_m3 = dense_bn(units=n_ch_0*3, name_to_concat='noise_2_mul_m3', input_tensor=noise_1_add_f)

    # Modulation of Noise layers - Addition++
    noise_2_add_m2 = dense_bn(units=n_ch_0*2, name_to_concat='noise_2_add_m2', input_tensor=noise_1_add_f)
    # Modulation of Noise layers - Multiplication++
    noise_2_mul_m2 = dense_bn(units=n_ch_0*2, name_to_concat='noise_2_mul_m2', input_tensor=noise_1_add_f)

    # Modulation of Noise layers - Addition+
    noise_2_add_m1 = dense_bn(units=n_ch_0*1, name_to_concat='noise_2_add_m1', input_tensor=noise_1_add_f)
    # Modulation of Noise layers - Multiplication+
    noise_2_mul_m1 = dense_bn(units=n_ch_0*1, name_to_concat='noise_2_mul_m1', input_tensor=noise_1_add_f)

    # Modulation of Noise layers - Addition
    noise_2_add = dense_bn(units=n_ch_0*4, name_to_concat='noise_2_add', input_tensor=noise_1_add_f)
    # Modulation of Noise layers - Multiplication
    noise_2_mul = dense_bn(units=n_ch_0*4, name_to_concat='noise_2_mul', input_tensor=noise_1_add_f)

    # Modulation of Noise layers - Addition+++
    noise_2_add_p3 = dense_bn(units=n_ch_0*3, name_to_concat='noise_2_add_p3', input_tensor=noise_1_add_f)
    # Modulation of Noise layers - Multiplication+++
    noise_2_mul_p3 = dense_bn(units=n_ch_0*3, name_to_concat='noise_2_mul_p3', input_tensor=noise_1_add_f)

    # Modulation of Noise layers - Addition++
    noise_2_add_p2 = dense_bn(units=n_ch_0*2, name_to_concat='noise_2_add_p2', input_tensor=noise_1_add_f)
    # Modulation of Noise layers - Multiplication++
    noise_2_mul_p2 = dense_bn(units=n_ch_0*2, name_to_concat='noise_2_mul_p2', input_tensor=noise_1_add_f)

    # Modulation of Noise layers - Addition+
    noise_2_add_p1 = dense_bn(units=n_ch_0*1, name_to_concat='noise_2_add_p1', input_tensor=noise_1_add_f)
    # Modulation of Noise layers - Multiplication+
    noise_2_mul_p1 = dense_bn(units=n_ch_0*1, name_to_concat='noise_2_mul_p1', input_tensor=noise_1_add_f)
    
    # Encoder
    conv_0 = conv2d_bn_relu(filter_size=3, filter_number=n_ch_0, name_to_concat='gen_0', input_tensor=inputs)
    conv_0 = Dropout(0.25, name='do_gen_a3')(conv_0)
    # M2 - Noise
    conv_0_noise = conv2d_bn(filter_size=3, filter_number=n_ch_0, name_to_concat='gen_noise_m1', input_tensor=conv_0)
    conv_0_noise = Dropout(0.25, name='do_gen_b3')(conv_0_noise)
    conv_0_noise = multiply([conv_0_noise, noise_2_mul_m1], name='mul_noiseZ_m1')
    conv_0_noise = add([conv_0_noise, noise_2_add_m1], name='add_noiseZ_m1')
    conv_0_noise = Activation('relu', name=str('relu_noise_m1'))(conv_0_noise)
    # M2 - Noise - Addition
    conv_0 = add([conv_0_noise, conv_0], name='add_noiseZres_m1')
    conv_1 = conv2d_bn_relu(filter_size=3, filter_number=n_ch_0, name_to_concat='gen_1', input_tensor=conv_0)
    pool_0 = MaxPooling2D(pool_size=(2, 2), name='maxpool2d_gen_0')(conv_1)

    conv_2 = conv2d_bn_relu(filter_size=3, filter_number=n_ch_0*2, name_to_concat='gen_2', input_tensor=pool_0)
    conv_2 = Dropout(0.25, name='do_gen_a2')(conv_2)
    # M2 - Noise
    conv_2_noise = conv2d_bn(filter_size=3, filter_number=n_ch_0*2, name_to_concat='gen_noise_m2', input_tensor=conv_2)
    conv_2_noise = Dropout(0.25, name='do_gen_b2')(conv_2_noise)
    conv_2_noise = multiply([conv_2_noise, noise_2_mul_m2], name='mul_noiseZ_m2')
    conv_2_noise = add([conv_2_noise, noise_2_add_m2], name='add_noiseZ_m2')
    conv_2_noise = Activation('relu', name=str('relu_noise_m2'))(conv_2_noise)
    # M2 - Noise - Addition
    conv_2 = add([conv_2_noise, conv_2], name='add_noiseZres_m2')
    conv_3 = conv2d_bn_relu(filter_size=3, filter_number=n_ch_0*2, name_to_concat='gen_3', input_tensor=conv_2)
    pool_1 = MaxPooling2D(pool_size=(2, 2), name='maxpool2d_gen_1')(conv_3)

    conv_4 = conv2d_bn_relu(filter_size=3, filter_number=n_ch_0*3, name_to_concat='gen_4', input_tensor=pool_1)
    conv_4 = Dropout(0.25, name='do_gen_a1')(conv_4)
    # M3 - Noise
    conv_4_noise = conv2d_bn(filter_size=3, filter_number=n_ch_0*3, name_to_concat='gen_noise_m3', input_tensor=conv_4)
    conv_4_noise = Dropout(0.25, name='do_gen_b1')(conv_4_noise)
    conv_4_noise = multiply([conv_4_noise, noise_2_mul_m3], name='mul_noiseZ_m3')
    conv_4_noise = add([conv_4_noise, noise_2_add_m3], name='add_noiseZ_m3')
    conv_4_noise = Activation('relu', name=str('relu_noise_m3'))(conv_4_noise)
    # M3 - Noise - Addition
    conv_4 = add([conv_4_noise, conv_4], name='add_noiseZres_m3')
    conv_5 = conv2d_bn_relu(filter_size=3, filter_number=n_ch_0*3, name_to_concat='gen_5', input_tensor=conv_4)
    pool_2 = MaxPooling2D(pool_size=(2, 2), name='maxpool2d_gen_2')(conv_5)
    
    # Bottleneck
    conv_6 = conv2d_bn_relu(filter_size=3, filter_number=n_ch_0*4, name_to_concat='gen_8', input_tensor=pool_2)
    conv_6 = Dropout(0.25, name='do_gen_0a')(conv_6)
    # Bottleneck - Noise
    conv_6_noise = conv2d_bn(filter_size=3, filter_number=n_ch_0*4, name_to_concat='gen_noise_p4', input_tensor=conv_6)
    conv_6_noise = Dropout(0.25, name='do_gen_0b')(conv_6_noise)
    conv_6_noise = multiply([conv_6_noise, noise_2_mul], name='mul_noiseZ_p4')
    conv_6_noise = add([conv_6_noise, noise_2_add], name='add_noiseZ_p4')
    conv_6_noise = Activation('relu', name=str('relu_noise_p4'))(conv_6_noise)
    # Bottleneck - Noise - Addition
    conv_6 = add([conv_6_noise, conv_6], name='add_noiseZres_p4')
    conv_7 = conv2d_bn_relu(filter_size=3, filter_number=n_ch_0*4, name_to_concat='gen_9', input_tensor=conv_6)
    pool_3 = deconv2d_bn_relu(filter_size=2, filter_number=n_ch_0*4, name_to_concat='de_gen_9', input_tensor=conv_7)
    pool_3 = concatenate([pool_3, conv_5], axis=-1, name='concat_gen_0')

    # Decoder
    conv_10 = conv2d_bn_relu(filter_size=3, filter_number=n_ch_0*3, name_to_concat='gen_10', input_tensor=pool_3)
    conv_10 = Dropout(0.25, name='do_gen_1a')(conv_10)
    # P3 - Noise
    conv_10_noise = conv2d_bn(filter_size=3, filter_number=n_ch_0*3, name_to_concat='gen_noise_p3', input_tensor=conv_10)
    conv_10_noise = Dropout(0.25, name='do_gen_1b')(conv_10_noise)
    conv_10_noise = multiply([conv_10_noise, noise_2_mul_p3], name='mul_noiseZ_p3')
    conv_10_noise = add([conv_10_noise, noise_2_add_p3], name='add_noiseZ_p3')
    conv_10_noise = Activation('relu', name=str('relu_noise_p3'))(conv_10_noise)
    # P3 - Noise - Addition
    conv_10 = add([conv_10_noise, conv_10], name='add_noiseZres_p3')
    conv_11 = conv2d_bn_relu(filter_size=3, filter_number=n_ch_0*3, name_to_concat='gen_11', input_tensor=conv_10)
    pool_5 = deconv2d_bn_relu(filter_size=2, filter_number=n_ch_0*3, name_to_concat='de_gen_11', input_tensor=conv_11)
    pool_5 = concatenate([pool_5, conv_3], axis=-1, name='concat_gen_1')

    conv_14 = conv2d_bn_relu(filter_size=3, filter_number=n_ch_0*2, name_to_concat='gen_14', input_tensor=pool_5)
    conv_14 = Dropout(0.25, name='do_gen_2a')(conv_14)
    # P2 - Noise
    conv_14_noise = conv2d_bn(filter_size=3, filter_number=n_ch_0*2, name_to_concat='gen_noise_p2', input_tensor=conv_14)
    conv_14_noise = Dropout(0.25, name='do_gen_2b')(conv_14_noise)
    conv_14_noise = multiply([conv_14_noise, noise_2_mul_p2], name='mul_noiseZ_p2')
    conv_14_noise = add([conv_14_noise, noise_2_add_p2], name='add_noiseZ_p2')
    conv_14_noise = Activation('relu', name=str('relu_noise_p2'))(conv_14_noise)
    # P2 - Noise - Addition
    conv_14 = add([conv_14_noise, conv_14], name='add_noiseZres_p2')
    conv_15 = conv2d_bn_relu(filter_size=3, filter_number=n_ch_0*2, name_to_concat='gen_15', input_tensor=conv_14)
    pool_7 = deconv2d_bn_relu(filter_size=2, filter_number=n_ch_0*2, name_to_concat='de_gen_15', input_tensor=conv_15)
    pool_7 = concatenate([pool_7, conv_1], axis=-1, name='concat_gen_3')

    conv_16 = conv2d_bn_relu(filter_size=3, filter_number=n_ch_0, name_to_concat='gen_16', input_tensor=pool_7)
    conv_16 = Dropout(0.25, name='do_gen_3a')(conv_16)
    # P2 - Noise
    conv_16_noise = conv2d_bn(filter_size=3, filter_number=n_ch_0*1, name_to_concat='gen_noise_p1', input_tensor=conv_16)
    conv_16_noise = Dropout(0.25, name='do_gen_3b')(conv_16_noise)
    conv_16_noise = multiply([conv_16_noise, noise_2_mul_p1], name='mul_noiseZ_p1')
    conv_16_noise = add([conv_16_noise, noise_2_add_p1], name='add_noiseZ_p1')
    conv_16_noise = Activation('relu', name=str('relu_noise_p1'))(conv_16_noise)
    # P2 - Noise - Addition
    conv_16 = add([conv_16_noise, conv_16], name='add_noiseZres_p1')
    conv_17 = conv2d_bn_relu(filter_size=3, filter_number=n_ch_0, name_to_concat='gen_17', input_tensor=conv_16)

    # Segmentation Layer
    segment = Conv2D(nc_out, (1, 1), padding='same', name='gen_segmentation')(conv_17)
    segment = Activation('tanh', name='non_lin_segment')(segment)

    model = Model(inputs=[inputs,inputNoiseZ], outputs=segment, name='Gen_UNet2D')
    return model


''' SECTION 5: Training 4 different networks in 4-fold
##
# '''


for fold in [1,2,3,4]:
    # Flush the GPU memory
    K.get_session().run(tf.global_variables_initializer())    
    logger = Logger('./logdir/'+save_file_name+'_fold'+str(fold))
    logger.log_graph(sess)

    # Network parameters
    first_fm_G = 32 # feature maps' size of the first layer 
    Diters = 5 # critic's learning steps in 1 step generator learning
    delta = 10 # WGAN-GP's delta learning parameters
    niter = 200 # Number of Generator's learning iterations

    imageSize = 256
    noiseSize = 32
    batchSize = 16

    lrD = 1e-4
    lrG = 1e-4

    gen_iterations = 0
    crit_iterations = 0
    crit_dem_iterations = 0
    errG = 0

    # In[8]:
    netD_y2 = Dis_C2D_FCN1((imageSize, imageSize, 1))
    netD_y2.summary()

    netD_dem = Dis_C2D_FCN1((imageSize, imageSize, 1))
    netD_dem.summary()

    # In[9]:
    netG = Gen_UNet2D((imageSize, imageSize, 2), (noiseSize, 1), first_fm_G, 1)
    netG.summary()

    # compute Wasserstein loss and  gradient penalty
    noiseZ = Input(shape=(noiseSize, 1))
    netD_real_input = Input(shape=(imageSize, imageSize, 1))
    netG_real_input = Input(shape=(imageSize, imageSize, 2))

    net_G_real_IM = Lambda(lambda x : x[:,:,:,0])(netG_real_input)
    net_G_real_IM = Reshape((imageSize, imageSize, 1))(net_G_real_IM)
    real_attribution = netD_real_input - net_G_real_IM

    ## Critic for Y2
    attribution = netG([netG_real_input, noiseZ])
    netD_y2_fake_input = net_G_real_IM + attribution

    ep_input = K.placeholder(shape=(None,1,1,1))
    netD_y2_mixed_input = Input(shape=(imageSize, imageSize, 1),
        tensor=ep_input * netD_real_input + (1-ep_input) * netD_y2_fake_input)

    loss_real = K.mean(netD_y2(netD_real_input))
    loss_fake = K.mean(netD_y2(netD_y2_fake_input))

    grad_mixed = K.gradients(netD_y2(netD_y2_mixed_input), [netD_y2_mixed_input])[0]
    norm_grad_mixed = K.sqrt(K.sum(K.square(grad_mixed), axis=[1,2,3]))
    grad_penalty = K.mean(K.square(norm_grad_mixed -1))

    loss = loss_fake - loss_real + delta * grad_penalty

    training_updates = Adam(lr=lrD, beta_1=0.0, beta_2=0.9).get_updates(netD_y2.trainable_weights,[],loss)
    netD_y2_train = K.function([netD_real_input, netG_real_input, noiseZ, ep_input],
                            [loss_real, loss_fake],    
                            training_updates)

    ## Critic for DEM
    ep_input_dem = K.placeholder(shape=(None,1,1,1))
    netD_mixed_input_dem = Input(shape=(imageSize, imageSize, 1),
        tensor=ep_input_dem * real_attribution + (1-ep_input_dem) * attribution)

    loss_real_dem = K.mean(netD_dem(real_attribution))
    loss_fake_dem = K.mean(netD_dem(attribution))

    grad_mixed_dem = K.gradients(netD_dem(netD_mixed_input_dem), [netD_mixed_input_dem])[0]
    norm_grad_mixed_dem = K.sqrt(K.sum(K.square(grad_mixed_dem), axis=[1,2,3]))
    grad_penalty_dem = K.mean(K.square(norm_grad_mixed_dem -1))

    loss_dem = loss_fake_dem - loss_real_dem + delta * grad_penalty_dem

    training_updates_dem = Adam(lr=lrD, beta_1=0.0, beta_2=0.9).get_updates(netD_dem.trainable_weights,[],loss_dem)
    netD_dem_train = K.function([netD_real_input, netG_real_input, noiseZ, ep_input_dem],
                            [loss_real_dem, loss_fake_dem],    
                            training_updates_dem)

    ## Loss for Generator
    delta_M1 = 100.0
    delta_M2 = 100.0
    loss_delta_M1 = K.mean(K.abs(attribution - real_attribution)) * delta_M1 # r4: (1-MSE)

    ## Additional Loss for Generator (WMH Volume Based)
    # WMH for DSC loss
    delta_M4 = 1.0 # r1: 1.0, r2: 10.0, r3: 100.0
    wmh_real_y2 = K.cast(K.greater_equal(netD_real_input, IM_TRSH), tf.float32)
    wmh_fake_y2 = K.cast(K.greater_equal(netD_y2_fake_input, IM_TRSH), tf.float32)
    loss_delta_M4 = dice_coef_loss(wmh_real_y2,wmh_fake_y2) * delta_M4 # dice_coef_loss

    # MSE of WMH volume * Constant
    delta_M3 = 100.0
    wmh_vol_real_y2 = K.sum(wmh_real_y2) / 1000.0
    wmh_vol_fake_y2 = K.sum(wmh_fake_y2) / 1000.0
    loss_delta_M3 = K.mean(K.square(wmh_vol_real_y2 - wmh_vol_fake_y2)) * delta_M3 # r4: (1-MSE)

    ## All losses combined
    loss = (-loss_fake) + (-loss_fake_dem) + loss_delta_M1 + loss_delta_M3 + loss_delta_M4 # r4: (-MSE)

    training_updates = Adam(lr=lrG, beta_1=0.0, beta_2=0.9).get_updates(netG.trainable_weights,[], loss)
    netG_no_update = K.function([netG_real_input, netD_real_input, noiseZ],
                                [loss, loss_fake, loss_fake_dem, loss_delta_M1, loss_delta_M3, loss_delta_M4])
    netG_train = K.function([netG_real_input, netD_real_input, noiseZ],
                                [loss, loss_fake, loss_fake_dem, loss_delta_M1, loss_delta_M3, loss_delta_M4], training_updates)

    ''' ---- LOAD TRAINING DATA 
    '''

    ### NOTE:
    ## Please change the .txt files' names as you wish.
    ## You also can change probability map into irregularity map
    ### Acronym:
    ## - 1tp      : 1st time point
    ## - 2tp      : 2nd time point
    ## - wmh_prob : WMH's probability/irregularity map
    ## - icv      : IntraCranial Volume
    ## - sl       : Stroke Lesions

    print("Reading data: FLAIR 1tp")
    data_list_flair_1tp = []
    f = open('./'+config_dir+'/flair_1tp_fold'+str(fold)+'.txt',"r")
    for line in f:
        data_list_flair_1tp.append(line)
    data_list_flair_1tp = map(lambda s: s.strip('\n'), data_list_flair_1tp)

    print("Reading data: PROB 1tp")
    data_list_prob_1tp = []
    f = open('./'+config_dir+'/wmh_prob_1tp_fold'+str(fold)+'.txt',"r")
    for line in f:
        data_list_prob_1tp.append(line)
    data_list_prob_1tp = map(lambda s: s.strip('\n'), data_list_prob_1tp)

    print("Reading data: PROB 2tp")
    data_list_prob_2tp = []
    f = open('./'+config_dir+'/wmh_prob_2tp_fold'+str(fold)+'.txt',"r")
    for line in f:
        data_list_prob_2tp.append(line)
    data_list_prob_2tp = map(lambda s: s.strip('\n'), data_list_prob_2tp)

    print("Reading data: ICV 1tp")
    data_list_icv_1tp = []
    f = open('./'+config_dir+'/icv_1tp_fold'+str(fold)+'.txt',"r")
    for line in f:
        data_list_icv_1tp.append(line)
    data_list_icv_1tp = map(lambda s: s.strip('\n'), data_list_icv_1tp)

    print("Reading data: ICV 2tp")
    data_list_icv_2tp = []
    f = open('./'+config_dir+'/icv_2tp_fold'+str(fold)+'.txt',"r")
    for line in f:
        data_list_icv_2tp.append(line)
    data_list_icv_2tp = map(lambda s: s.strip('\n'), data_list_icv_2tp)

    print("Reading data: SL 1tp")
    data_list_sl_1tp = []
    f = open('./'+config_dir+'/sl_cleaned_1tp_fold'+str(fold)+'.txt',"r")
    for line in f:
        data_list_sl_1tp.append(line)
    data_list_sl_1tp = map(lambda s: s.strip('\n'), data_list_sl_1tp)

    print("Reading data: SL 2tp")
    data_list_sl_2tp = []
    f = open('./'+config_dir+'/sl_cleaned_2tp_fold'+str(fold)+'.txt',"r")
    for line in f:
        data_list_sl_2tp.append(line)
    data_list_sl_2tp = map(lambda s: s.strip('\n'), data_list_sl_2tp)

    id = 0
    train_prob_flair_1tp = np.zeros((1,256,256,2))
    train_prob_2tp = np.zeros((1,256,256,1))
    for data in data_list_prob_1tp:
        if os.path.isfile(data):
            print(data)
            print(data_list_prob_2tp[id])
            print(data_list_flair_1tp[id])
            print(data_list_icv_1tp[id])
            print(data_list_icv_2tp[id])
            print(data_list_wmh_1tp[id])
            print(data_list_wmh_2tp[id])

            loaded_data_p_1tp = load_data(data)
            loaded_data_p_2tp = load_data(data_list_prob_2tp[id])
            loaded_data_f_1tp = load_data(data_list_flair_1tp[id])
            loaded_data_i_1tp = load_data(data_list_icv_1tp[id])
            loaded_data_i_2tp = load_data(data_list_icv_2tp[id])
            loaded_data_w_1tp = load_data(data_list_wmh_1tp[id])
            loaded_data_w_2tp = load_data(data_list_wmh_2tp[id])

            loaded_image_p_1tp = data_prep(loaded_data_p_1tp)
            loaded_image_p_2tp = data_prep(loaded_data_p_2tp)
            loaded_image_f_1tp = data_prep(loaded_data_f_1tp)
            loaded_image_i_1tp = data_prep(loaded_data_i_1tp)
            loaded_image_i_2tp = data_prep(loaded_data_i_2tp)
            loaded_image_w_1tp = data_prep(loaded_data_w_1tp)
            loaded_image_w_2tp = data_prep(loaded_data_w_2tp)

            brain_prob_1tp = np.multiply(loaded_image_p_1tp, loaded_image_i_1tp)
            brain_flair_1tp = np.multiply(loaded_image_f_1tp, loaded_image_i_1tp)
            brain_prob_2tp = np.multiply(loaded_image_p_2tp, loaded_image_i_2tp)
            
            if os.path.isfile(data_list_sl_1tp[id]):
                print(data_list_sl_1tp[id])
                loaded_data_s_1tp  = load_data(data_list_sl_1tp[id])
                loaded_image_s_1tp = data_prep(loaded_data_s_1tp)
                loaded_image_s_1tp = 1 - loaded_image_s_1tp
                brain_prob_1tp = np.multiply(brain_prob_1tp, loaded_image_s_1tp)
                brain_flair_1tp = np.multiply(brain_flair_1tp, loaded_image_s_1tp)
                
            if os.path.isfile(data_list_sl_2tp[id]):
                print(data_list_sl_2tp[id])
                loaded_data_s_2tp  = load_data(data_list_sl_2tp[id])
                loaded_image_s_2tp = data_prep(loaded_data_s_2tp)
                loaded_image_s_2tp = 1 - loaded_image_s_2tp
                brain_prob_2tp = np.multiply(brain_prob_2tp, loaded_image_s_2tp)
            
            norm_percentile = 0
            print "MRI 1tp [old] - max: " + str(np.max(brain_flair_1tp)) + ", min: " + str(np.min(brain_flair_1tp))
            brain_flair_1tp = map_image_to_intensity_range(brain_flair_1tp, 0, 1, percentiles=norm_percentile)
            print "MRI 1tp [new] - max: " + str(np.max(brain_flair_1tp)) + ", min: " + str(np.min(brain_flair_1tp))
            
            print "MRI 1tp [old] - max: " + str(np.max(brain_prob_1tp)) + ", min: " + str(np.min(brain_prob_1tp))
            print "MRI 1tp [new] - max: " + str(np.max(brain_prob_1tp)) + ", min: " + str(np.min(brain_prob_1tp))

            print "MRI 2tp [old] - max: " + str(np.max(brain_prob_2tp)) + ", min: " + str(np.min(brain_prob_2tp))
            print "MRI 1tp [new] - max: " + str(np.max(brain_prob_2tp)) + ", min: " + str(np.min(brain_prob_2tp))

            brain_prob_1tp[brain_prob_1tp < 0] = 0
            brain_prob_2tp[brain_prob_2tp < 0] = 0
            
            # Concatenate probability map with flair of the 1st time point
            brain_prob_1tp = np.concatenate((brain_prob_1tp, brain_flair_1tp), axis=-1)
            print brain_prob_1tp.shape
            print brain_prob_2tp.shape
            
            print("ALL LOADED")
            if id == 0:
                train_prob_flair_1tp = brain_prob_1tp
                train_prob_2tp = brain_prob_2tp
            else:
                train_prob_flair_1tp = np.concatenate((train_prob_flair_1tp,brain_prob_1tp), axis=0)
                train_prob_2tp = np.concatenate((train_prob_2tp,brain_prob_2tp), axis=0)
                print "train_prob_flair_1tp SHAPE: " + str(train_prob_flair_1tp.shape) + " | " + str(id)
            id += 1

    print("train_prob_flair_1tp -> ", train_prob_flair_1tp.shape)
    print("train_prob_2tp -> ", train_prob_2tp.shape)

    ''' ---- Split and shuffle the LOADED TRAINING DATA 
    '''
    ## Split the data for training and validation
    prob_flair_1tp_train, prob_flair_1tp_val, prob_2tp_train, prob_2tp_val = train_test_split(train_prob_flair_1tp, train_prob_2tp, test_size=0.02, random_state=42)

    ## Check the size of training and validation data
    print("prob_flair_1tp_train -> ", prob_flair_1tp_train.shape)
    print("prob_2tp_train -> ", prob_2tp_train.shape)
    print("prob_flair_1tp_val -> ", prob_flair_1tp_val.shape)
    print("prob_2tp_val -> ", prob_2tp_val.shape)

    ## Check the range values of the training and validation data 
    print("prob_flair_1tp_train - max: " + str(np.max(prob_flair_1tp_train)) + ", min: " + str(np.min(prob_flair_1tp_train)))
    print("prob_2tp_train - max: " + str(np.max(prob_2tp_train)) + ", min: " + str(np.min(prob_2tp_train)))
    print("prob_flair_1tp_val - max: " + str(np.max(prob_flair_1tp_val)) + ", min: " + str(np.min(prob_flair_1tp_val)))
    print("prob_2tp_val - max: " + str(np.max(prob_2tp_val)) + ", min: " + str(np.min(prob_2tp_val)))

    ## Shuffle the training data
    indices = np.array(range(prob_flair_1tp_train.shape[0]))
    np.random.shuffle(indices)
    prob_flair_1tp_train = prob_flair_1tp_train[indices]
    prob_2tp_train = prob_2tp_train[indices]

    # Shuffle the validation data
    indices = np.array(range(prob_flair_1tp_val.shape[0]))
    np.random.shuffle(indices)
    prob_flair_1tp_val = prob_flair_1tp_val[indices]
    prob_2tp_val = prob_2tp_val[indices]

    # Create a fixed noise array for validation
    fixed_noise = np.random.normal(size=(prob_flair_1tp_val.shape[0], noiseSize, 1)).astype('float32')

    # Log the validation image
    logger.log_images('val_img_Y1_IM', prob_flair_1tp_val[:50,:,:,0], gen_iterations, '')
    logger.log_images('val_img_Y2_IM', prob_2tp_val[:50], gen_iterations, '')

    vn, vx, vy, vc = prob_flair_1tp_val.shape

    ''' Generator's learning iterations
    '''
    t0 = time.time() # timer
    for epoch in range(niter):
        i = 0
        ii = 0
        indices = np.array(range(prob_flair_1tp_train.shape[0]))
        np.random.shuffle(indices)

        prob_flair_1tp_train = prob_flair_1tp_train[indices]
        prob_2tp_train = prob_2tp_train[indices]
            
        batches = prob_flair_1tp_train.shape[0]//batchSize
        
        while i < batches:
            if gen_iterations < 25 or gen_iterations % 500 == 0:
                _Diters = 100
                _Diters_dem = 100
            else:
                _Diters = Diters
                _Diters_dem = Diters
            j = 0
            jj =0

            ## Train the Critic Y2
            while j < _Diters and i < batches:
                j+=1
                real_data_1tp = prob_flair_1tp_train[i*batchSize:(i+1)*batchSize]
                real_data_2tp = prob_2tp_train[i*batchSize:(i+1)*batchSize]
                i+=1
                noise = np.random.normal(size=(batchSize, noiseSize, 1))
                ep = np.random.uniform(size=(batchSize, 1, 1 ,1))
                errD_real, errD_fake  = netD_y2_train([real_data_2tp, real_data_1tp, noise, ep])
                errD = errD_real - errD_fake
                logger.log_scalar('errCrit_aaLosses', errD, crit_iterations)
                logger.log_scalar('errCrit_aReal_losses', errD_real, crit_iterations)
                logger.log_scalar('errCrit_aFake_losses', errD_fake, crit_iterations)
                crit_iterations += 1
            
            ## Train the Critic DEM
            while jj < _Diters_dem and ii < batches:
                jj+=1
                real_data_1tp = prob_flair_1tp_train[ii*batchSize:(ii+1)*batchSize]
                real_data_2tp = prob_2tp_train[ii*batchSize:(ii+1)*batchSize]
                ii+=1
                noise = np.random.normal(size=(batchSize, noiseSize, 1))
                ep = np.random.uniform(size=(batchSize, 1, 1 ,1))
                errD_real_dem, errD_fake_dem  = netD_dem_train([real_data_2tp, real_data_1tp, noise, ep])
                errD_dem = errD_real_dem - errD_fake_dem
                logger.log_scalar('errCrit_DEM_aaLosses', errD_dem, crit_dem_iterations)
                logger.log_scalar('errCrit_DEM_aReal_losses', errD_real_dem, crit_dem_iterations)
                logger.log_scalar('errCrit_DEM_aFake_losses', errD_fake_dem, crit_dem_iterations)
                crit_dem_iterations += 1

            ## Save the Critic Y2's training error to Tensorboard
            logger.log_scalar('errDC_aaLosses', errD, gen_iterations)
            logger.log_scalar('errDC_aReal_losses', errD_real, gen_iterations)
            logger.log_scalar('errDC_aFake_losses', errD_fake, gen_iterations)

            ## Save the Critic DEM's training error to Tensorboard
            logger.log_scalar('errDC_DEM_aaLosses', errD_dem, gen_iterations)
            logger.log_scalar('errDC_DEM_aReal_losses', errD_real_dem, gen_iterations)
            logger.log_scalar('errDC_DEM_aFake_losses', errD_fake_dem, gen_iterations)
            
            ## Do VALIDATION
            if gen_iterations%10==0:
                print('TRN FOLD %d - [%d/%d][%d/%d][%d] Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f'
                % (fold, epoch, niter, i, batches, gen_iterations, errD, errG, errD_real, errD_fake), time.time()-t0)
                
                val_D_fake_loss = np.mean(netD_y2.predict(np.reshape(prob_flair_1tp_val[:,:,:,0], (vn, vx, vy, 1))))
                val_D_real_loss = np.mean(netD_y2.predict(prob_2tp_val))
                val_D_real_generated_loss = np.mean(netD_y2.predict(netG.predict([prob_flair_1tp_val,fixed_noise])))
                
                print('VAL FOLD %d - [%d/%d][%d/%d][%d] Loss_D_fake: %f Loss_D_real: %f Loss_D_real_gen: %f'
                % (fold, epoch, niter, i, batches, gen_iterations, val_D_fake_loss, val_D_real_loss, val_D_real_generated_loss))
                
                logger.log_scalar('val_D_fake_loss', val_D_fake_loss, gen_iterations)
                logger.log_scalar('val_D_real_loss', val_D_real_loss, gen_iterations)
                logger.log_scalar('val_D_real_generated_loss', val_D_real_generated_loss, gen_iterations)

                ## Save VALIDATION Image using Tensorboard
                if gen_iterations%500==0:                
                    attributed = netG.predict([prob_flair_1tp_val,fixed_noise])
                    fake = np.reshape(prob_flair_1tp_val[:,:,:,0], (vn, vx, vy, 1)) + attributed
                    
                    logger.log_images('attributed_img_step%d' % (gen_iterations),
                                        attributed[:50], gen_iterations, '')
                    logger.log_images('fake_img_step%d' % (gen_iterations),
                                        fake[:50], gen_iterations, '')
                    
            ## Train the Generator
            k_noise = 10
            losses_errG = []
            noises = np.random.normal(size=(k_noise, batchSize, noiseSize, 1)).astype('float32')
            for k in range(k_noise):
                noise = noises[k]
                errG,errG_CY2,errG_DEM,errG_MSE,errG_VOL,errG_WMH = netG_no_update([real_data_1tp, real_data_2tp, noise]) # v3
                losses_errG.append(errG)
            numpy_losses_errG = np.array(losses_errG)
            losses_errG_argmin = numpy_losses_errG.argmin(0)
            noise = noises[losses_errG_argmin]
            errG,errG_CY2,errG_DEM,errG_MSE,errG_VOL,errG_WMH = netG_train([real_data_1tp, real_data_2tp, noise]) # v3

            ## Save the Generator's training error
            logger.log_scalar('errG_losses', errG, gen_iterations)
            logger.log_scalar('errG_CY2_losses', errG_CY2, gen_iterations)
            logger.log_scalar('errG_DEM_losses', errG_DEM, gen_iterations)
            logger.log_scalar('errG_MSE_losses', errG_MSE, gen_iterations)
            logger.log_scalar('errG_VOL_losses', errG_VOL, gen_iterations)
            logger.log_scalar('errG_WMH_losses', errG_WMH, gen_iterations)

            print('GEN ERR FOLD %d - [%d/%d][%d/%d][%d] errG: %f errG_CY2: %f errG_DEM: %f errG_MSE: %f errG_VOL: %f errG_WMH: %f'
            % (fold, epoch, niter, i, batches, gen_iterations, errG, errG_CY2, errG_DEM, errG_MSE, errG_VOL, errG_WMH))

            ## Save models
            netG.save('./models/netG_'+save_file_name+'_fold'+str(fold)+'.h5')

            gen_iterations+=1 

