import tensorflow as tf
import keras
from keras.layers import Conv3D
from keras.layers import BatchNormalization
from deformable_conv3d import *

class DCNN3D(Conv3D):
    """DCNN3D
    Convolutional layer responsible for learning the 3D offsets and output the
    deformed feature map using bilinear interpolation
    and then do further convolution
    Note that this layer does not perform convolution on the deformed feature
    map.
    """

    def __init__(self, nb_batch, num_outputs, kernel_size, scope, norm=True, d_format='NDHWC', **kwargs):
    # def __init__(self, ):
        """Init
        """
        self.nb_batch = nb_batch
        self.num_outputs = num_outputs
        self.kernel_size1 = kernel_size
        self.scope = scope
        self.norm = norm
        self.d_format = d_format
        super(DCNN3D, self).__init__(
            self.kernel_size1[0] * self.kernel_size1[1] * self.kernel_size1[2] * 3, (3, 3, 3), padding='same', use_bias=False,
            kernel_initializer=keras.initializers.Zeros(),
            **kwargs
        )

    def build(self, input_shape):
        self.kernel11 = self.add_weight(name='kernel11',
                                      shape=(self.kernel_size1[0], self.kernel_size1[1], self.kernel_size1[2], input_shape[4], self.num_outputs),
                                      initializer='uniform',
                                      trainable=True)
        super(DCNN3D, self).build(input_shape)

    def call(self, x):
        """Return the deformed featured map"""
        # generate offset-field
        # offset = tf.contrib.layers.conv2d(
        #     self.inputs, self.kernel_size1[0] * self.kernel_size1[1] * 2, [3, 3], scope=self.scope + '/offset',
        #     data_format=self.d_format, activation_fn=None, weights_initializer=tf.zeros_initializer(dtype=tf.float32),
        #     biases_initializer=None)
        offset = super(DCNN3D, self).call(x)

        # BN
        # offset = tf.contrib.layers.batch_norm(
        #     offset, decay=0.9, center=True, activation_fn=tf.nn.tanh,
        #     updates_collections=None, epsilon=1e-5, scope=self.scope + '/offset' + '/batch_norm',
        #     data_format='NHWC')
        offset = tf.layers.batch_normalization(offset,trainable=False)
        offset = tf.nn.tanh(offset)

        # generate deformed feature
        input_shape = [self.nb_batch, x.shape[1].value, x.shape[2].value, x.shape[3].value, x.shape[4].value]
        dcn = DCN(input_shape, self.kernel_size1)
        deformed_feature = dcn.deform_conv(x, offset, self.scope)

        # return deformed_feature

        # conv on the deformed feature
        outputs = tf.nn.conv3d(deformed_feature, self.kernel11, strides=(1, self.kernel_size1[0], self.kernel_size1[1], self.kernel_size1[2], 1), padding="VALID")

        if self.norm:
            # outputs = tf.contrib.layers.batch_norm(
            #     outputs, decay=0.9, center=True, activation_fn=tf.nn.relu,
            #     updates_collections=None, epsilon=1e-5, scope=self.scope + '/batch_norm',
            #     data_format='NHWC')
            # outputs = BatchNormalization(axis=-1)(outputs)
            outputs = tf.layers.batch_normalization(outputs)
            outputs = tf.nn.relu(outputs, name=self.scope + '/relu')
        else:
            outputs = tf.nn.relu(outputs, name=self.scope + '/relu')
        return outputs

    def compute_output_shape(self, input_shape):
        """Output shape is the same as input shape except the num_channels
        """
        return (input_shape[0], input_shape[1], input_shape[2], input_shape[3], self.num_outputs)
        # return (input_shape[0], self.kernel_size1[0]*input_shape[1], self.kernel_size1[1]*input_shape[2], input_shape[3])