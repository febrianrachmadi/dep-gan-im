
# coding: utf-8

# In[1]:

''' SECTION 1: Some variables need to be specified
##
# '''

# This code is modifed from https://github.com/martinarjovsky/WassersteinGAN 
# get_ipython().magic(u'matplotlib inline')

import os
os.environ['KERAS_BACKEND']='tensorflow' # tensorflow
os.environ["CUDA_VISIBLE_DEVICES"]="0" # set the id number of GPU

## Specify where to save the trained model
save_file_name = 'depuresnet_pNoises_T02_19042019_fold'

## Specify the location of .txt files for accessing the training data
config_dir = 'train_data_server_fold'

''' SECTION 2: Call libraries
##
# '''

import keras
import keras.backend as K
# K.set_image_data_format('channels_first')
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

from StringIO import StringIO
import matplotlib.pyplot as plt

import nibabel as nib
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import tensorflow as tf

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
#     print image_data.shape
    image_data = np.squeeze(image_data)
    output_img = np.swapaxes(image_data, 0, 2)
    output_img = np.rot90(output_img)
    output_img = output_img[::-1,...]   # flipped

    return output_img

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

def convert_to_1hot(label, n_class):
    # Convert a label map (N x H x W x 1) into a one-hot representation (N x H x W x C)
    print " --> SIZE = " + str(label.shape)
    print " --> MAX = " + str(np.max(label))
    print " --> MIN = " + str(np.min(label))
    label_flat = label.flatten().astype(int)
    n_data = len(label_flat)
    print " --> SIZE = " + str(label_flat.shape)
    print " --> LEN = " + str(n_data)
    label_1hot = np.zeros((n_data, n_class), dtype='int16')
    print " --> 1HOT-SIZE = " + str(label_1hot.shape)
    label_1hot[range(n_data), label_flat] = 1
    label_1hot = label_1hot.reshape((label.shape[0], label.shape[1], label.shape[2], label.shape[3], n_class))

    return label_1hot

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


## DEP-UResNet network
def Gen_UNet2D(input_shape, noiseZ_shape=(32, 1), first_fm=32, nc_out=1):
    n_ch_0 = first_fm
    inputs = Input(input_shape, name='input_gen_chn_0')
    
    # Inputting the noises
    noiseZ_shape_batch = (noiseZ_shape[0], noiseZ_shape[1]) 
    inputNoiseZ = Input(noiseZ_shape_batch, name='input_gen_noiseZ_0')     

    # Modulation of Noise layers - Firsts
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
    # M2 - Noise
    conv_0_noise = conv2d_bn(filter_size=3, filter_number=n_ch_0, name_to_concat='gen_noise_m1', input_tensor=conv_0)
    conv_0_noise = multiply([conv_0_noise, noise_2_mul_m1], name='mul_noiseZ_m1') 
    conv_0_noise = add([conv_0_noise, noise_2_add_m1], name='add_noiseZ_m1') 
    conv_0_noise = Activation('relu', name=str('relu_noise_m1'))(conv_0_noise)
    # M2 - Noise - Addition
    conv_0 = add([conv_0_noise, conv_0], name='add_noiseZres_m1') 
    conv_1 = conv2d_bn_relu(filter_size=3, filter_number=n_ch_0, name_to_concat='gen_1', input_tensor=conv_0)
    pool_0 = MaxPooling2D(pool_size=(2, 2), name='maxpool2d_gen_0')(conv_1)

    conv_2 = conv2d_bn_relu(filter_size=3, filter_number=n_ch_0*2, name_to_concat='gen_2', input_tensor=pool_0)
    # M2 - Noise
    conv_2_noise = conv2d_bn(filter_size=3, filter_number=n_ch_0*2, name_to_concat='gen_noise_m2', input_tensor=conv_2)
    conv_2_noise = multiply([conv_2_noise, noise_2_mul_m2], name='mul_noiseZ_m2') 
    conv_2_noise = add([conv_2_noise, noise_2_add_m2], name='add_noiseZ_m2') 
    conv_2_noise = Activation('relu', name=str('relu_noise_m2'))(conv_2_noise)
    # M2 - Noise - Addition
    conv_2 = add([conv_2_noise, conv_2], name='add_noiseZres_m2') 
    conv_3 = conv2d_bn_relu(filter_size=3, filter_number=n_ch_0*2, name_to_concat='gen_3', input_tensor=conv_2)
    pool_1 = MaxPooling2D(pool_size=(2, 2), name='maxpool2d_gen_1')(conv_3)

    conv_4 = conv2d_bn_relu(filter_size=3, filter_number=n_ch_0*3, name_to_concat='gen_4', input_tensor=pool_1)
    # M3 - Noise
    conv_4_noise = conv2d_bn(filter_size=3, filter_number=n_ch_0*3, name_to_concat='gen_noise_m3', input_tensor=conv_4)
    conv_4_noise = multiply([conv_4_noise, noise_2_mul_m3], name='mul_noiseZ_m3') 
    conv_4_noise = add([conv_4_noise, noise_2_add_m3], name='add_noiseZ_m3') 
    conv_4_noise = Activation('relu', name=str('relu_noise_m3'))(conv_4_noise)
    # M3 - Noise - Addition
    conv_4 = add([conv_4_noise, conv_4], name='add_noiseZres_m3') 
    conv_5 = conv2d_bn_relu(filter_size=3, filter_number=n_ch_0*3, name_to_concat='gen_5', input_tensor=conv_4)
    pool_2 = MaxPooling2D(pool_size=(2, 2), name='maxpool2d_gen_2')(conv_5)
    
    # Bottleneck
    conv_6 = conv2d_bn_relu(filter_size=3, filter_number=n_ch_0*4, name_to_concat='gen_8', input_tensor=pool_2)
    # Bottleneck - Noise
    conv_6_noise = conv2d_bn(filter_size=3, filter_number=n_ch_0*4, name_to_concat='gen_noise_p4', input_tensor=conv_6)
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
    conv_10 = Dropout(0.25, name='do_gen_1')(conv_10)
    # P3 - Noise
    conv_10_noise = conv2d_bn(filter_size=3, filter_number=n_ch_0*3, name_to_concat='gen_noise_p3', input_tensor=conv_10)
    conv_10_noise = multiply([conv_10_noise, noise_2_mul_p3], name='mul_noiseZ_p3') 
    conv_10_noise = add([conv_10_noise, noise_2_add_p3], name='add_noiseZ_p3') 
    conv_10_noise = Activation('relu', name=str('relu_noise_p3'))(conv_10_noise)
    # P3 - Noise - Addition
    conv_10 = add([conv_10_noise, conv_10], name='add_noiseZres_p3') 
    conv_11 = conv2d_bn_relu(filter_size=3, filter_number=n_ch_0*3, name_to_concat='gen_11', input_tensor=conv_10)
    pool_5 = deconv2d_bn_relu(filter_size=2, filter_number=n_ch_0*3, name_to_concat='de_gen_11', input_tensor=conv_11)
    pool_5 = concatenate([pool_5, conv_3], axis=-1, name='concat_gen_1')

    conv_14 = conv2d_bn_relu(filter_size=3, filter_number=n_ch_0*2, name_to_concat='gen_14', input_tensor=pool_5)
    # P2 - Noise
    conv_14_noise = conv2d_bn(filter_size=3, filter_number=n_ch_0*2, name_to_concat='gen_noise_p2', input_tensor=conv_14)
    conv_14_noise = multiply([conv_14_noise, noise_2_mul_p2], name='mul_noiseZ_p2') 
    conv_14_noise = add([conv_14_noise, noise_2_add_p2], name='add_noiseZ_p2') 
    conv_14_noise = Activation('relu', name=str('relu_noise_p2'))(conv_14_noise)
    # P2 - Noise - Addition
    conv_14 = add([conv_14_noise, conv_14], name='add_noiseZres_p2') 
    conv_15 = conv2d_bn_relu(filter_size=3, filter_number=n_ch_0*2, name_to_concat='gen_15', input_tensor=conv_14)
    pool_7 = deconv2d_bn_relu(filter_size=2, filter_number=n_ch_0*2, name_to_concat='de_gen_15', input_tensor=conv_15)
    pool_7 = concatenate([pool_7, conv_1], axis=-1, name='concat_gen_3')

    conv_16 = conv2d_bn_relu(filter_size=3, filter_number=n_ch_0, name_to_concat='gen_16', input_tensor=pool_7)
    # P2 - Noise
    conv_16_noise = conv2d_bn(filter_size=3, filter_number=n_ch_0*1, name_to_concat='gen_noise_p1', input_tensor=conv_16)
    conv_16_noise = multiply([conv_16_noise, noise_2_mul_p1], name='mul_noiseZ_p1') 
    conv_16_noise = add([conv_16_noise, noise_2_add_p1], name='add_noiseZ_p1') 
    conv_16_noise = Activation('relu', name=str('relu_noise_p1'))(conv_16_noise)
    # P2 - Noise - Addition
    conv_16 = add([conv_16_noise, conv_16], name='add_noiseZres_p1') 
    conv_17 = conv2d_bn_relu(filter_size=3, filter_number=n_ch_0, name_to_concat='gen_17', input_tensor=conv_16)

    # Segmentation Layer
    segment = Conv2D(nc_out, (1, 1), padding='same', name='gen_segmentation')(conv_17)
    segment = Activation('softmax', name='non_lin_segment')(segment)

    model = Model(inputs=[inputs,inputNoiseZ], outputs=segment, name='Gen_UNet2D')
    model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy')
    return model

''' SECTION 5: Training 4 different networks in 4-fold
##
# '''

for fold in [1,2,3,4]:
    # Flush the GPU memory
    K.get_session().run(tf.global_variables_initializer())
    save_filename = save_file_name+str(fold)

    ### NOTE:
    ## Please change the .txt files' names as you wish.
    ### Acronym:
    ## - 1tp      : 1st time point
    ## - 2tp      : 2nd time point
    ## - icv      : IntraCranial Volume
    ## - sl       : Stroke Lesions

    print("Reading data: FLAIR 1tp")
    data_list_flair_1tp = []
    f = open('./'+config_dir+'/flair_1tp_fold'+str(fold)+'.txt',"r")
    for line in f:
        data_list_flair_1tp.append(line)
    data_list_flair_1tp = map(lambda s: s.strip('\n'), data_list_flair_1tp)

    print("Reading data: WMH subtracted coded")
    data_list_wsc_1tp = []
    f = open('./'+config_dir+'/wmh_subtracted_coded_2tp_1tp_fold'+str(fold)+'.txt',"r")
    for line in f:
        data_list_wsc_1tp.append(line)
    data_list_wsc_1tp = map(lambda s: s.strip('\n'), data_list_wsc_1tp)

    print("Reading data: ICV 1tp")
    data_list_icv_1tp = []
    f = open('./'+config_dir+'/icv_1tp_fold'+str(fold)+'.txt',"r")
    for line in f:
        data_list_icv_1tp.append(line)
    data_list_icv_1tp = map(lambda s: s.strip('\n'), data_list_icv_1tp)

    print("Reading data: SL 1tp")
    data_list_sl_1tp = []
    f = open('./'+config_dir+'/sl_cleaned_1tp_fold'+str(fold)+'.txt',"r")
    for line in f:
        data_list_sl_1tp.append(line)
    data_list_sl_1tp = map(lambda s: s.strip('\n'), data_list_sl_1tp)

    id = 0
    train_flair_1tp = np.zeros((1,256,256,1))
    train_wscod_1tp = np.zeros((1,256,256,1))
    for data in data_list_flair_1tp:
        if os.path.isfile(data):
            print(data)
            print(data_list_wsc_1tp[id])
            print(data_list_icv_1tp[id])
            print(data_list_sl_1tp[id])

            loaded_data_f_1tp = load_data(data)
            loaded_data_wsc_1tp = load_data(data_list_wsc_1tp[id])
            loaded_data_i_1tp = load_data(data_list_icv_1tp[id])

            loaded_image_f_1tp = data_prep(loaded_data_f_1tp)
            loaded_image_wsc_1tp = data_prep(loaded_data_wsc_1tp)
            loaded_image_i_1tp = data_prep(loaded_data_i_1tp)

            brain_flair_1tp = np.multiply(loaded_image_f_1tp, loaded_image_i_1tp)
            brain_wsc_1tp = np.multiply(loaded_image_wsc_1tp, loaded_image_i_1tp)
            
            if os.path.isfile(data_list_sl_1tp[id]):
                print(data_list_sl_1tp[id])
                loaded_data_s_1tp  = load_data(data_list_sl_1tp[id])
                loaded_image_s_1tp = data_prep(loaded_data_s_1tp)
                loaded_image_s_1tp = 1 - loaded_image_s_1tp
                brain_flair_1tp = np.multiply(brain_flair_1tp, loaded_image_s_1tp)
                brain_wsc_1tp = np.multiply(brain_wsc_1tp, loaded_image_s_1tp)
            
            print brain_flair_1tp.shape
            print brain_wsc_1tp.shape

            norm_percentile = 0
            print "WSC 1tp [new] - max: " + str(np.max(brain_wsc_1tp)) + ", min: " + str(np.min(brain_wsc_1tp))
            
            print "FLR 1tp [old] - mean: " + str(np.mean(brain_flair_1tp)) + ", std: " + str(np.std(brain_flair_1tp))
            brain_flair_1tp = ((brain_flair_1tp - np.mean(brain_flair_1tp)) / np.std(brain_flair_1tp)) # normalise to zero mean unit variance 3D
            brain_flair_1tp = np.nan_to_num(brain_flair_1tp)
            print "FLR 1tp [new] - mean: " + str(np.mean(brain_flair_1tp)) + ", std: " + str(np.std(brain_flair_1tp))

            print brain_flair_1tp.shape
            print brain_wsc_1tp.shape
            
            print("ALL LOADED")
            if id == 0:
                train_flair_1tp = brain_flair_1tp
                train_wscod_1tp = brain_wsc_1tp
            else:
                train_flair_1tp = np.concatenate((train_flair_1tp,brain_flair_1tp), axis=0)
                train_wscod_1tp = np.concatenate((train_wscod_1tp,brain_wsc_1tp), axis=0)
                print "train_flair_1tp SHAPE: " + str(train_flair_1tp.shape) + " | " + str(id)
            id += 1


    # In[17]:
    print("train_flair_1tp -> ", train_flair_1tp.shape)
    print("train_wscod_1tp -> ", train_wscod_1tp.shape)

    ''' ---- Split and shuffle the LOADED TRAINING DATA 
    '''
    ## Split the data for training and validation
    flair_1tp_train, flair_1tp_val, wscod_1tp_train, wscod_1tp_val = train_test_split(train_flair_1tp, train_wscod_1tp, test_size=0.02, random_state=42)

    ## Shuffle the training data
    indices = np.array(range(flair_1tp_train.shape[0]))
    np.random.shuffle(indices)
    flair_1tp_train = flair_1tp_train[indices]
    wscod_1tp_train = wscod_1tp_train[indices]

    # Shuffle the validation data
    indices = np.array(range(flair_1tp_val.shape[0]))
    np.random.shuffle(indices)
    flair_1tp_val = flair_1tp_val[indices]
    wscod_1tp_val = wscod_1tp_val[indices]

    ## Check the size of training and validation data
    print("flair_1tp_train -> ", flair_1tp_train.shape)
    print("wscod_1tp_train -> ", wscod_1tp_train.shape)
    print("flair_1tp_val -> ", flair_1tp_val.shape)
    print("wscod_1tp_val -> ", wscod_1tp_val.shape)

    ## Check the range values of the training and validation data 
    print("flair_1tp_train - mean: " + str(np.mean(flair_1tp_train)) + ", std: " + str(np.std(flair_1tp_train)))
    print("wscod_1tp_train - max: " + str(np.max(wscod_1tp_train)) + ", min: " + str(np.min(wscod_1tp_train)))
    print("flair_1tp_val - mean: " + str(np.mean(flair_1tp_val)) + ", std: " + str(np.std(flair_1tp_val)))
    print("wscod_1tp_val - max: " + str(np.max(wscod_1tp_val)) + ", min: " + str(np.min(wscod_1tp_val)))

    print("Convert to 1-hot representation:")
    wscod_1tp_train = wscod_1tp_train.astype(int)
    wscod_1tp_val = wscod_1tp_val.astype(int)
    wscod_1tp_train_label = convert_to_1hot(wscod_1tp_train, 4)
    wscod_1tp_val_label = convert_to_1hot(wscod_1tp_val, 4)
    wscod_1tp_train_label = np.squeeze(wscod_1tp_train_label)
    wscod_1tp_val_label = np.squeeze(wscod_1tp_val_label)
    print("wscod_1tp_train_label -> ", wscod_1tp_train_label.shape)
    print("wscod_1tp_val_label -> ", wscod_1tp_val_label.shape)

    # Network parameters
    n_label = 4     # number of labels
    noiseSize = 32  # size of noises
    first_fm_G = 32 # feature maps' size of the first layer
    imageSize = 256

    ## Create the DEP-UResNet
    my_network = Gen_UNet2D((imageSize, imageSize, 1), (noiseSize, 1), first_fm_G, n_label)
    my_network.summary()

    print 'Training..'
    # early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)
    # history_batch = LossHistory()

    my_network.summary()
    my_network.get_config()
    keras.backend.get_session().run(tf.global_variables_initializer())

    nb_samples    = 16
    num_epochs    = 200
    bn_momentum   = 0.99
    shuffle_epoch = True

    # Create a fixed noise array for validation
    fixed_noise = np.random.normal(size=(flair_1tp_val.shape[0], noiseSize, 1)).astype('float32')

    ''' Training iterations
    '''
    for ep in range(num_epochs):
        # Print iteration's information
        print "\n---\nEPOCH using FOR-IN: " + str(ep+1) + "/" + str(int(num_epochs))

        # Create random noises
        noise = np.random.normal(size=(flair_1tp_train.shape[0], noiseSize, 1)).astype('float32')
        history_callback = my_network.fit([flair_1tp_train, noise], wscod_1tp_train_label,
            epochs=1,
            batch_size=nb_samples,
            shuffle=shuffle_epoch,
            validation_data=([flair_1tp_val, fixed_noise], wscod_1tp_val_label))
        
        ## SAVE LOSS HISTORY
        loss_history = history_callback.history["loss"]
        numpy_loss_history = np.array(loss_history)
        f = open('./logs/loss_'+save_filename+'.txt', 'ab')
        np.savetxt(f,numpy_loss_history)

        ## SAVE VAL LOSS HISTORY
        loss_history = history_callback.history["val_loss"]
        numpy_loss_history = np.array(loss_history)
        f = open('./logs/val_loss_'+save_filename+'.txt', 'ab')
        np.savetxt(f,numpy_loss_history)

        ## SAVING TRAINED MODEL AND WEIGHTS
        ## Save models
        my_network.save('./models/trained_'+save_filename+'.h5')
        model_json = my_network.to_json()
        with open('./models/model_'+save_filename+'.json', "w") as json_file:
            json_file.write(model_json)

