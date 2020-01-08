
# coding: utf-8

''' SECTION 1: Some variables need to be specified
##
# '''
import h5py, os, math
os.environ['KERAS_BACKEND']='tensorflow' # tensorflow
os.environ["CUDA_VISIBLE_DEVICES"]="0"

## Specify where the trained model
saved_model_name = 'netG_depgan_twoCritics_prob_noSL_21102019_fold'

## Specify where the output file
dirOutputPath = '/mnt/HDD/febrian/Results_GAN/'
saving_filename_dir = 'DEP-GANs-PROB-FLAIR-MEAN-RUN10-21102019'

## Specify the location of .txt files for accessing the training data
config_dir = 'test_data_fold'

''' SECTION 2: Call libraries
##
# '''

import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Conv2D, ZeroPadding2D, BatchNormalization, Input, Dropout, Reshape, Layer
from keras.layers import UpSampling2D, MaxPooling2D, AveragePooling2D, Dense, Lambda, Conv1D
from keras.layers import Conv2DTranspose, Reshape, Activation, Cropping2D, Flatten, Add, LeakyReLU
from keras.layers.merge import concatenate, multiply, add
from keras.optimizers import RMSprop, SGD, Adam
from keras.activations import relu
from keras.initializers import RandomNormal
conv_init = RandomNormal(0, 0.02)
gamma_init = RandomNormal(1., 0.02)

import numpy as np
from sklearn.model_selection import train_test_split

import nibabel as nib
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
        self.pixdim = nim.header['pixdim'][1:4]

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


''' SECTION 5: Testing 4 different networks in 4-fold
##
# '''

# ---- CREATE RESULT DIRs
dirOutput = dirOutputPath + saving_filename_dir
print(dirOutput)
try:
    os.makedirs(dirOutput)
except OSError:
    if not os.path.isdir(dirOutput):
        raise
print(dirOutput)

vol_dsc_best_all = []
for fold in [1,2,3,4]:

    # Parameters
    first_fm_G = 32  # feature maps' size of the first layer 
    imageSize = 256
    noiseSize = 32

    lrD = 1e-4
    lrG = 1e-4

    TRSH_VAL = 0.5

    netG = Gen_UNet2D((imageSize, imageSize, 2), (noiseSize, 1), first_fm_G, 1)
    netG.summary()

    netG.load_weights('./models/' + saved_model_name + str(fold) + '.h5')
    print("Generator's weights loaded!")

    ### NOTE:
    ## Please change the .txt files' names as you wish.
    ## You also can change probability map into irregularity map
    ### Acronym:
    ## - 1tp      : 1st time point
    ## - 2tp      : 2nd time point
    ## - wmh_prob : WMH's probability/irregularity map
    ## - icv      : IntraCranial Volume
    ## - sl       : Stroke Lesions

    # ---- LOAD TRAINING DATA
    print("Reading data: FLAIR 1tp")
    data_list_flair_1tp = []
    f = open('./'+config_dir+'/flair_1tp_fold'+str(fold)+'.txt',"r")
    for line in f:
        data_list_flair_1tp.append(line)
    data_list_flair_1tp = list(map(lambda s: s.strip('\n'), data_list_flair_1tp))

    print("Reading data: PROB v2 1tp")
    data_list_prob_1tp = []
    f = open('./'+config_dir+'/wmh_prob_v2_1tp_fold'+str(fold)+'.txt',"r")
    for line in f:
        data_list_prob_1tp.append(line)
    data_list_prob_1tp = list(map(lambda s: s.strip('\n'), data_list_prob_1tp))

    print("Reading data: IM 1tp")
    data_list_im_1tp = []
    f = open('./'+config_dir+'/iam_1tp_fold'+str(fold)+'.txt',"r")
    for line in f:
        data_list_im_1tp.append(line)
    data_list_im_1tp = list(map(lambda s: s.strip('\n'), data_list_im_1tp))

    print("Reading data: PROB v2 2tp")
    data_list_prob_2tp = []
    f = open('./'+config_dir+'/wmh_prob_v2_2tp_fold'+str(fold)+'.txt',"r")
    for line in f:
        data_list_prob_2tp.append(line)
    data_list_prob_2tp = list(map(lambda s: s.strip('\n'), data_list_prob_2tp))

    print("Reading data: ICV 1tp")
    data_list_icv_1tp = []
    f = open('./'+config_dir+'/icv_1tp_fold'+str(fold)+'.txt',"r")
    for line in f:
        data_list_icv_1tp.append(line)
    data_list_icv_1tp = list(map(lambda s: s.strip('\n'), data_list_icv_1tp))

    print("Reading data: WMH 1tp")
    data_list_wmh_1tp = []
    f = open('./'+config_dir+'/wmh_1tp_fold'+str(fold)+'.txt',"r")
    for line in f:
        data_list_wmh_1tp.append(line)
    data_list_wmh_1tp = list(map(lambda s: s.strip('\n'), data_list_wmh_1tp))

    print("Reading data: SL 1tp")
    data_list_sl_1tp = []
    f = open('./'+config_dir+'/sl_cleaned_1tp_fold'+str(fold)+'.txt',"r")
    for line in f:
        data_list_sl_1tp.append(line)
    data_list_sl_1tp = list(map(lambda s: s.strip('\n'), data_list_sl_1tp))

    print("Reading data: WMH 2tp")
    data_list_wmh_2tp = []
    f = open('./'+config_dir+'/wmh_2tp_fold'+str(fold)+'.txt',"r")
    for line in f:
        data_list_wmh_2tp.append(line)
    data_list_wmh_2tp = list(map(lambda s: s.strip('\n'), data_list_wmh_2tp))

    print("Reading data: WMH evolution coded")
    data_list_code_2tp = []
    f = open('./'+config_dir+'/wmh_subtracted_coded_2tp_1tp_fold'+str(fold)+'.txt',"r")
    for line in f:
        data_list_code_2tp.append(line)
    data_list_code_2tp = list(map(lambda s: s.strip('\n'), data_list_code_2tp))

    print("Reading data: ICV 2tp")
    data_list_icv_2tp = []
    f = open('./'+config_dir+'/icv_2tp_fold'+str(fold)+'.txt',"r")
    for line in f:
        data_list_icv_2tp.append(line)
    data_list_icv_2tp = list(map(lambda s: s.strip('\n'), data_list_icv_2tp))

    print("Reading data: SL 2tp")
    data_list_sl_2tp = []
    f = open('./'+config_dir+'/sl_cleaned_2tp_fold'+str(fold)+'.txt',"r")
    for line in f:
        data_list_sl_2tp.append(line)
    data_list_sl_2tp = list(map(lambda s: s.strip('\n'), data_list_sl_2tp))

    print("Reading data: NAME")
    data_list_name = []
    f = open('./'+config_dir+'/name_fold'+str(fold)+'.txt',"r")
    for line in f:
        data_list_name.append(line)
    data_list_name = list(map(lambda s: s.strip('\n'), data_list_name))

    id = 0
    train_flair_1tp = np.zeros((1,256,256,1))
    train_iam___1tp = np.zeros((1,256,256,1))
    for data in data_list_flair_1tp:
        if os.path.isfile(data):

            # Print data location
            print(data_list_name[id])
            print(data)
            print(data_list_im_1tp[id])
            print(data_list_prob_1tp[id])
            print(data_list_prob_2tp[id])
            print(data_list_icv_1tp[id])
            print(data_list_sl_1tp[id])
            print(data_list_wmh_1tp[id])
            print(data_list_wmh_2tp[id])
            print(data_list_code_2tp[id])
            print(data_list_icv_2tp[id])
            print(data_list_sl_2tp[id])

            # Load nifti data
            loaded_data_f_1tp = load_data(data)
            loaded_data_im_1tp = load_data(data_list_im_1tp[id])
            loaded_data_p_1tp = load_data(data_list_prob_1tp[id])
            loaded_data_p_2tp = load_data(data_list_prob_2tp[id])
            loaded_data_i_1tp = load_data(data_list_icv_1tp[id])
            loaded_data_w_1tp = load_data(data_list_wmh_1tp[id])
            loaded_data_w_2tp = load_data(data_list_wmh_2tp[id])
            loaded_data_i_2tp = load_data(data_list_icv_2tp[id])
            loaded_data_c_2tp = load_data(data_list_code_2tp[id])

            # Prepare the loaded nifti data
            loaded_image_f_1tp = data_prep(loaded_data_f_1tp)
            loaded_image_im_1tp = data_prep(loaded_data_im_1tp)
            loaded_image_p_1tp = data_prep(loaded_data_p_1tp)
            loaded_image_p_2tp = data_prep(loaded_data_p_2tp)
            loaded_image_i_1tp = data_prep(loaded_data_i_1tp)
            loaded_image_w_1tp = data_prep(loaded_data_w_1tp)
            loaded_image_w_2tp = data_prep(loaded_data_w_2tp)
            loaded_image_i_2tp = data_prep(loaded_data_i_2tp)
            loaded_image_c_2tp = data_prep(loaded_data_c_2tp)

            loaded_image_f_1tp = np.squeeze(loaded_image_f_1tp)
            loaded_image_im_1tp = np.squeeze(loaded_image_im_1tp)
            loaded_image_p_1tp = np.squeeze(loaded_image_p_1tp)
            loaded_image_p_2tp = np.squeeze(loaded_image_p_2tp)
            loaded_image_i_1tp = np.squeeze(loaded_image_i_1tp)
            loaded_image_w_1tp = np.squeeze(loaded_image_w_1tp)
            loaded_image_w_2tp = np.squeeze(loaded_image_w_2tp)
            loaded_image_i_2tp = np.squeeze(loaded_image_i_2tp)
            loaded_image_c_2tp = np.squeeze(loaded_image_c_2tp)

            # Exclude non-brain tissues
            brain_flair_1tp = np.multiply(loaded_image_f_1tp, loaded_image_i_1tp)
            brain_im____1tp = np.multiply(loaded_image_im_1tp, loaded_image_i_1tp)
            brain_prob__1tp = np.multiply(loaded_image_p_1tp, loaded_image_i_1tp)
            brain_prob__2tp = np.multiply(loaded_image_p_2tp, loaded_image_i_2tp)
            brain_wmh_1tp = np.multiply(loaded_image_w_1tp, loaded_image_i_1tp)
            brain_wmh_2tp = np.multiply(loaded_image_w_2tp, loaded_image_i_2tp)
            brain_code_2tp = np.multiply(loaded_image_c_2tp, loaded_image_i_2tp)              
            
            # Exclude Stroke Lesions (SL) tissues on 1st time point data
            stroke_seg_count = 0
            icv_and_sl_mask_1tp = loaded_image_i_1tp
            if os.path.isfile(data_list_sl_1tp[id]):
                loaded_data_s_1tp  = load_data(data_list_sl_1tp[id])
                loaded_image_s_1tp = data_prep(loaded_data_s_1tp)
                loaded_image_s_1tp = np.squeeze(loaded_image_s_1tp)
                stroke_seg_count = np.count_nonzero(loaded_image_s_1tp)
                loaded_image_s_1tp = 1 - loaded_image_s_1tp
                brain_prob__1tp = np.multiply(brain_prob__1tp, loaded_image_s_1tp)
                brain_im____1tp = np.multiply(brain_im____1tp, loaded_image_s_1tp)
                brain_flair_1tp = np.multiply(brain_flair_1tp, loaded_image_s_1tp)
                brain_wmh_1tp = np.multiply(brain_wmh_1tp, loaded_image_s_1tp)
                icv_and_sl_mask_1tp = np.multiply(icv_and_sl_mask_1tp, loaded_image_s_1tp)
                        
            # Exclude Stroke Lesions (SL) tissues on 2nd time point data
            icv_and_sl_mask_2tp = loaded_image_i_2tp
            if os.path.isfile(data_list_sl_2tp[id]):
                loaded_data_s_2tp  = load_data(data_list_sl_2tp[id])
                loaded_image_s_2tp = data_prep(loaded_data_s_2tp)
                loaded_image_s_2tp = np.squeeze(loaded_image_s_2tp)
                loaded_image_s_2tp = 1 - loaded_image_s_2tp
                brain_wmh_2tp = np.multiply(brain_wmh_2tp, loaded_image_s_2tp)
                brain_prob__2tp = np.multiply(brain_prob__2tp, loaded_image_s_2tp)
                icv_and_sl_mask_2tp = np.multiply(loaded_image_i_2tp, loaded_image_s_2tp)

            # Check how many WMH based on the probability map
            # Note that wmh_mask will be replaced later by manual mask of WMH 
            wmh_mask = np.zeros(brain_prob__1tp.shape)
            wmh_mask = brain_prob__1tp >= TRSH_VAL
            wmh_seg_count = np.count_nonzero(wmh_mask)
            print("WMH count: " + str(wmh_seg_count))
            
            # Normalise FLAIR values
            norm_percentile = 0
            print("FLAIR 1tp [old] - max: " + str(np.max(brain_flair_1tp)) + ", min: " + str(np.min(brain_flair_1tp)))
            brain_flair_1tp = map_image_to_intensity_range(brain_flair_1tp, 0, 1, percentiles=norm_percentile)
            print("FLAIR 1tp [new] - max: " + str(np.max(brain_flair_1tp)) + ", min: " + str(np.min(brain_flair_1tp)))

            # Exclude IM and PM's values below 0
            brain_im____1tp[brain_im____1tp < 0] = 0
            brain_prob__1tp[brain_prob__1tp < 0] = 0
            brain_prob__2tp[brain_prob__2tp < 0] = 0

            print("PROB 1tp [old] - max: " + str(np.max(brain_prob__1tp)) + ", min: " + str(np.min(brain_prob__1tp)))
            print("PROB 1tp [new] - max: " + str(np.max(brain_prob__1tp)) + ", min: " + str(np.min(brain_prob__1tp)))
            
            print("IM 1tp [old] - max: " + str(np.max(brain_im____1tp)) + ", min: " + str(np.min(brain_im____1tp)))
            print("IM 1tp [new] - max: " + str(np.max(brain_im____1tp)) + ", min: " + str(np.min(brain_im____1tp)))

            sx, sy, sz = brain_prob__1tp.shape
            brain_im____1tp = np.reshape(brain_im____1tp, (sx, sy, sz, 1))
            brain_prob__1tp = np.reshape(brain_prob__1tp, (sx, sy, sz, 1))
            brain_flair_1tp = np.reshape(brain_flair_1tp, (sx, sy, sz, 1))
            wmh_mask = np.reshape(wmh_mask, (sx, sy, sz, 1))

            print("brain_im____1tp -> ", brain_im____1tp.shape)
            print("brain_prob__1tp -> ", brain_prob__1tp.shape)
            print("brain_flair_1tp -> ", brain_flair_1tp.shape)
            print("wmh_mask -> ", wmh_mask.shape)
            
            brain_prob__1tp = np.concatenate((brain_prob__1tp, brain_flair_1tp), axis=-1)   # uncomment to use PM as input
            # brain_prob__1tp = np.concatenate((brain_im____1tp, brain_flair_1tp), axis=-1)   # uncomment to use IM as input
            print("brain_prob__1tp -> ", brain_prob__1tp.shape)

            # Produce 10 results by using 10 different sets of noise
            n_repeat = 10
            output_img_pred_mean = np.zeros(icv_and_sl_mask_2tp.shape)
            for repeat in range(n_repeat):
                print("REPEAT >>> " + str(repeat) + " - START <<<")
                noise = np.random.normal(size=(42, noiseSize, 1)).astype('float32')
                output_img_pred = netG.predict([brain_prob__1tp,noise])
                output_img_pred = np.squeeze(output_img_pred)
                output_img_pred = np.multiply(output_img_pred, icv_and_sl_mask_2tp) # clean using ICV and WMH
                output_img_pred_mean = output_img_pred_mean + output_img_pred
                print "REPEAT >>> " + str(repeat) + " - END <<<\n"

            # Evaluate the mean map of all 10 results
            output_img_pred = output_img_pred_mean / float(n_repeat)

            ''' Starting from here, the codes are used to evaluate all metrics used in the manuscript.
            '''

            ## Evaluate the "Volumetric Changes" in ml
            print("")
            print("WMH volume in the 1st time point")
            print("non zero (icv_and_sl_mask_1tp): ", np.count_nonzero(icv_and_sl_mask_1tp))
            wmh_mask = brain_wmh_1tp
            wmh_from_iam_1tp = np.multiply(icv_and_sl_mask_1tp, wmh_mask)
            print("ONE (wmh_from_iam_1tp): ", np.count_nonzero(wmh_from_iam_1tp))
            vol_1tp_mm3 = np.count_nonzero(wmh_from_iam_1tp) * np.prod(loaded_data_f_1tp.pixdim)
            vol_1tp__ml = vol_1tp_mm3 / 1000
            print("VOL (vol_1tp__ml): ", vol_1tp__ml)
            print("")
            
            print("WMH volume in the 2nd time point")
            print("non zero (icv_and_sl_mask_2tp): ", np.count_nonzero(icv_and_sl_mask_2tp))
            wmh_mask = brain_wmh_2tp
            wmh_from_iam_2tp = np.multiply(icv_and_sl_mask_2tp, wmh_mask)
            print("ONE (wmh_from_iam_2tp): ", np.count_nonzero(wmh_from_iam_2tp))
            vol_2tp_mm3 = np.count_nonzero(wmh_from_iam_2tp) * np.prod(loaded_data_f_1tp.pixdim)
            vol_2tp__ml = vol_2tp_mm3 / 1000
            print("VOL (vol_2tp__ml): ", vol_2tp__ml)
            print("")
            
            print("WMH volume of 1tp data from IM")
            wmh_mask = np.zeros(brain_prob__1tp.shape)
            wmh_mask = brain_prob__1tp >= TRSH_VAL
            print("ONE (wmh_from_iam_1tp): ", np.count_nonzero(wmh_mask))
            vol_1tp_mm3_iam = np.count_nonzero(wmh_mask) * np.prod(loaded_data_f_1tp.pixdim)
            vol_1tp__ml_iam = vol_1tp_mm3_iam / 1000
            print("VOL (vol_1tp_ml_im): ", vol_1tp__ml_iam)
            print("")

            print("WMH volume of 2tp data from IM")
            wmh_mask = np.zeros(brain_prob__2tp.shape)
            wmh_mask = np.copy(brain_prob__2tp) >= TRSH_VAL
            print("ONE (wmh_from_iam_2tp): ", np.count_nonzero(wmh_mask))
            vol_2tp_mm3_iam = np.count_nonzero(wmh_mask) * np.prod(loaded_data_f_1tp.pixdim)
            vol_2tp__ml_iam = vol_2tp_mm3_iam / 1000
            print("VOL (vol_2tp_ml_im): ", vol_2tp__ml_iam)
            print("")
            
            print("OUTPUT (predicted) WMH volume of the 2nd time point")
            print("non zero (icv_and_sl_mask_2tp): ", np.count_nonzero(icv_and_sl_mask_2tp))
            brain_prob__2tp_fake = brain_prob__1tp[:,:,:,0] + output_img_pred
            brain_prob__2tp_fake[brain_prob__2tp_fake < -1] = -1
            brain_prob__2tp_fake[brain_prob__2tp_fake > 1] = 1
            wmh_mask = np.zeros(brain_prob__2tp_fake.shape)
            wmh_mask[brain_prob__2tp_fake > TRSH_VAL] = 1
            wmh_from_out_2tp = wmh_mask            
            wmh_from_out_2tp = np.multiply(icv_and_sl_mask_2tp, wmh_from_out_2tp)
            print("ONE (wmh_from_out_2tp): ", np.count_nonzero(wmh_from_out_2tp))
            vol_out_mm3 = np.count_nonzero(wmh_from_out_2tp) * np.prod(loaded_data_f_1tp.pixdim)
            vol_out__ml = vol_out_mm3 / 1000
            print("VOL (vol_out__ml): ", vol_out__ml)
            print("")

            err_vol = vol_out__ml - vol_2tp__ml
            mse_vol = np.mean((vol_2tp__ml - vol_out__ml)**2)
            
            # If the volume of WMH is GROWING or SHRINKING
            true_pred = 0
            true_prog = 0
            true_regg = 0
            prog = 0
            regg = 0
            if (vol_2tp__ml - vol_1tp__ml) >= 0:
                prog = 1
                if vol_out__ml - vol_1tp__ml >= 0:
                    true_pred = 1
                    true_prog = 1
            else: ## If WMH is SHRINKING
                regg = 1
                if vol_out__ml - vol_1tp__ml < 0:
                    true_pred = 1
                    true_regg = 1

            print("[PREDICTIONS, MSE_VOL] - WMH: " + str(
                [true_pred, prog, true_prog, regg, true_regg, mse_vol, err_vol]))

            ## Spatial Dynamic WMH evolution
            print(" ---> WMH Change Mask (Fake)")
            wmh_change__2tp_fake = np.squeeze(np.copy(brain_prob__2tp_fake))
            wmh_change_mask_fake = np.squeeze(np.zeros(brain_code_2tp.shape))
            wmh_change_mask_real = np.squeeze(brain_code_2tp)
            prob_1tp = np.copy(brain_prob__1tp[:,:,:,0])
            prob_1tp = np.squeeze(prob_1tp)

            print(str(wmh_change_mask_fake.shape))
            print(str(wmh_change_mask_real.shape))
            
            print("(Cat #1 - SHRINK)")
            mask1 = wmh_change__2tp_fake < TRSH_VAL
            mask3 = prob_1tp >= TRSH_VAL
            mask_fake = np.all([mask1,mask3], axis=0)
            print("Non zero FAKE (Cat #1 - SHRINK): ", np.count_nonzero(mask_fake))
            wmh_change_mask_fake[mask_fake] = 1

            print("(Cat #2 - GROW)")
            mask1 = wmh_change__2tp_fake >= TRSH_VAL
            mask3 = prob_1tp < TRSH_VAL
            mask_fake = np.all([mask1,mask3], axis=0)
            print("Non zero FAKE (Cat #2 - GROW): ", np.count_nonzero(mask_fake))
            wmh_change_mask_fake[mask_fake] = 2
            
            print("(Cat #3 - STAY)")
            mask1 = wmh_change__2tp_fake >= TRSH_VAL
            mask3 = prob_1tp >= TRSH_VAL
            mask_fake = np.all([mask1,mask3], axis=0)
            print("Non zero FAKE (Cat #3 - STAY): ", np.count_nonzero(mask_fake))
            wmh_change_mask_fake[mask_fake] = 3
            
            smooth = 1e-7
            # dice_1: Dice for Shrinking WMH
            k = 1
            dice_1 = (np.count_nonzero(wmh_change_mask_fake[wmh_change_mask_real == k] == k)*2.0 + smooth) / \
                        (smooth + np.count_nonzero(wmh_change_mask_real[wmh_change_mask_real == k] == k) + \
                        np.count_nonzero(wmh_change_mask_fake[wmh_change_mask_fake == k] == k))
            # dice_2: Dice for Growing WMH
            k = 2
            dice_2 = (np.count_nonzero(wmh_change_mask_fake[wmh_change_mask_real == k] == k)*2.0 + smooth) / \
                        (smooth + np.count_nonzero(wmh_change_mask_real[wmh_change_mask_real == k] == k) + \
                        np.count_nonzero(wmh_change_mask_fake[wmh_change_mask_fake == k] == k))        
            # dice_3: Dice for Stable WMH
            k = 3
            dice_3 = (np.count_nonzero(wmh_change_mask_fake[wmh_change_mask_real == k] == k)*2.0 + smooth) / \
                        (smooth + np.count_nonzero(wmh_change_mask_real[wmh_change_mask_real == k] == k) + \
                        np.count_nonzero(wmh_change_mask_fake[wmh_change_mask_fake == k] == k))  

            wmh_mask_fake = np.zeros(wmh_change_mask_fake.shape)
            wmh_mask_real = np.zeros(wmh_change_mask_fake.shape)
            wmh_mask_fake = wmh_change_mask_fake > 0
            wmh_mask_real = wmh_change_mask_real > 0

            # dice_4: Dice for the Whole WMH
            k = 1
            dice_4 = (np.count_nonzero(wmh_mask_fake[wmh_mask_real == k] == k)*2.0 + smooth) / \
                        (smooth + np.count_nonzero(wmh_mask_real[wmh_mask_real == k] == k) + \
                        np.count_nonzero(wmh_mask_fake[wmh_mask_fake == k] == k))

            wmh_mask_fake = np.zeros(wmh_change_mask_fake.shape)
            wmh_mask_real = np.zeros(wmh_change_mask_fake.shape)
            temp_1_fake = wmh_change_mask_fake == 1
            temp_1_real = wmh_change_mask_real == 1
            temp_2_fake = wmh_change_mask_fake == 2
            temp_2_real = wmh_change_mask_real == 2
            temp_a_fake = temp_1_fake + temp_2_fake
            temp_a_real = temp_1_real + temp_2_real
            wmh_mask_fake = temp_a_fake > 0
            wmh_mask_real = temp_a_real > 0

            # dice_5: Dice for Changing WMH (i.e., shrinking and growing)
            k = 1
            dice_5 = (np.count_nonzero(wmh_mask_fake[wmh_mask_real == k] == k)*2.0 + smooth) / \
                        (smooth + np.count_nonzero(wmh_mask_real[wmh_mask_real == k] == k) + \
                        np.count_nonzero(wmh_mask_fake[wmh_mask_fake == k] == k))

            wmh_mask_fake = np.zeros(wmh_change_mask_fake.shape)
            wmh_mask_real = np.zeros(wmh_change_mask_fake.shape)
            wmh_mask_fake = wmh_change_mask_fake == 3
            wmh_mask_real = wmh_change_mask_real == 3

            # dice_6: Dice for Stable WMH
            k = 1
            dice_6 = (np.count_nonzero(wmh_mask_fake[wmh_mask_real == k] == k)*2.0 + smooth) / \
                        (smooth + np.count_nonzero(wmh_mask_real[wmh_mask_real == k] == k) + \
                        np.count_nonzero(wmh_mask_fake[wmh_mask_fake == k] == k))
            
            print('DSC of Cat #1 (SHRINK) : {}'.format(dice_1))
            print('DSC of Cat #2 (GROW)   : {}'.format(dice_2))
            print('DSC of Cat #3 (STABLE) : {}'.format(dice_3))
            print('DSC of Cat #4 (WMH)    : {}'.format(dice_4))

            avg_all_dice = (dice_1 + dice_2 + dice_3) / 3.0 # Average
            avg_dice__56 = (dice_5 + dice_6) / 2.0  # Average
            vol_dsc  = [true_pred, prog, true_prog, regg, true_regg, vol_1tp__ml, vol_2tp__ml, vol_out__ml, 
                mse_vol, err_vol, dice_5, dice_6, avg_dice__56, dice_1, dice_2, dice_3, dice_4, avg_all_dice]
            vol_dsc_best_all.append(np.array(vol_dsc))

            print("DICE metrics #1 - WMH: " + str([dice_1, dice_2, dice_3, dice_4, avg_all_dice]))
            print("DICE metrics #2 - WMH: " + str([dice_5, dice_6, avg_dice__56]))
            
            brain_prob__2tp_fake = brain_prob__1tp[:,:,:,0] + output_img_pred
            brain_prob__2tp_fake[brain_prob__2tp_fake < -1] = -1
            brain_prob__2tp_fake[brain_prob__2tp_fake > 1] = 1
            output_img_ok = data_prep_save(brain_prob__2tp_fake)
            print("Saving file..")
            nim = nib.Nifti1Image(output_img_ok.astype('float32'), loaded_data_f_1tp.affine)
            print(dirOutput + '/' + data_list_name[id] + '.nii.gz')
            nib.save(nim, dirOutput + '/' + data_list_name[id] + '_2tp_prob_fake.nii.gz')
            
            output_img_ok = data_prep_save(output_img_pred)
            print("Saving file..")
            nim = nib.Nifti1Image(output_img_ok.astype('float32'), loaded_data_f_1tp.affine)
            print(dirOutput + '/' + data_list_name[id] + '.nii.gz')
            nib.save(nim, dirOutput + '/' + data_list_name[id] + '_network_output.nii.gz')

            output_img_ok = data_prep_save(wmh_change_mask_fake)
            print("Saving file..")
            nim = nib.Nifti1Image(output_img_ok.astype('float32'), loaded_data_f_1tp.affine)
            print(dirOutput + '/' + data_list_name[id] + '.nii.gz')
            nib.save(nim, dirOutput + '/' + data_list_name[id] + '_2tp_code_fake.nii.gz')

            id += 1
        else:
            print("Could not find: " + data)

        ## Save all evaluations to .csv file
        numpy_vol_dsc_best_all = np.array(vol_dsc_best_all)
        f = open(dirOutput + '/RECAP_evaluation_for_allData.csv', 'w')
        np.savetxt(f, numpy_vol_dsc_best_all, delimiter=",")
        f.close()
