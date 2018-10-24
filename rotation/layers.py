from keras.layers.core import Layer
import keras
import keras.backend as K
import tensorflow as tf
import numpy as np
#import transformations.transformer as transformer

class SpatialTransformer(Layer):
    """Spatial Transformer Layer
    Implements a spatial transformer layer as described in [1]_.
    Borrowed from [2]_:
    References
    ----------
    .. [1]  Spatial Transformer Networks
            Max Jaderberg, Karen Simonyan, Andrew Zisserman, Koray Kavukcuoglu
            Submitted on 5 Jun 2015
    .. [2]  https://github.com/skaae/transformer_network/blob/master/transformerlayer.py

    .. [3]  https://github.com/EderSantana/seya/blob/keras1/seya/layers/attention.py
    """

    def __init__(self,
                 localization_net,
                 output_size,
                 **kwargs):
        self.locnet = localization_net
        self.output_size = output_size
        super(SpatialTransformer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.locnet.build(input_shape)
        self.trainable_weights = self.locnet.trainable_weights
#         self.regularizers = self.locnet.regularizers 
#         self.constraints = self.locnet.constraints

    def compute_output_shape(self, input_shape):
        output_size = self.output_size
        return (None,
                int(output_size[0]),
                int(output_size[1]),
                int(input_shape[-1]))

    def call(self, X, mask=None):
        affine_transformation = self.locnet.call(X)
        output = self._transform(affine_transformation, X, self.output_size)
        #output=transformer(X,affine_transformation,self.output_size)
        return output

    def _repeat(self, x, num_repeats):
        ones = tf.ones((1, num_repeats), dtype='int32')
        x = tf.reshape(x, shape=(-1,1))
        x = tf.matmul(x, ones)
        return tf.reshape(x, [-1])

    def _interpolate(self, image, x, y, output_size):
        batch_size = tf.shape(image)[0]
        height = tf.shape(image)[1]
        width = tf.shape(image)[2]
        num_channels = tf.shape(image)[3]

        x = tf.cast(x , dtype='float32')
        y = tf.cast(y , dtype='float32')

        height_float = tf.cast(height, dtype='float32')
        width_float = tf.cast(width, dtype='float32')

        output_height = output_size[0]
        output_width  = output_size[1]

        x = .5*(x + 1.0)*(width_float)
        y = .5*(y + 1.0)*(height_float)

        x0 = tf.cast(tf.floor(x), 'int32')
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), 'int32')
        y1 = y0 + 1

        max_y = tf.cast(height - 1, dtype='int32')
        max_x = tf.cast(width - 1,  dtype='int32')
        zero = tf.zeros([], dtype='int32')

        x0 = tf.clip_by_value(x0, zero, max_x)
        x1 = tf.clip_by_value(x1, zero, max_x)
        y0 = tf.clip_by_value(y0, zero, max_y)
        y1 = tf.clip_by_value(y1, zero, max_y)

        flat_image_dimensions = width*height
        pixels_batch = tf.range(batch_size)*flat_image_dimensions
        flat_output_dimensions = output_height*output_width
        base = self._repeat(pixels_batch, flat_output_dimensions)
        base_y0 = base + y0*width
        base_y1 = base + y1*width
        indices_a = base_y0 + x0
        indices_b = base_y1 + x0
        indices_c = base_y0 + x1
        indices_d = base_y1 + x1

        flat_image = tf.reshape(image, shape=(-1, num_channels))
        flat_image = tf.cast(flat_image, dtype='float32')
        pixel_values_a = tf.gather(flat_image, indices_a)
        pixel_values_b = tf.gather(flat_image, indices_b)
        pixel_values_c = tf.gather(flat_image, indices_c)
        pixel_values_d = tf.gather(flat_image, indices_d)

        x0 = tf.cast(x0, 'float32')
        x1 = tf.cast(x1, 'float32')
        y0 = tf.cast(y0, 'float32')
        y1 = tf.cast(y1, 'float32')

        area_a = tf.expand_dims(((x1 - x) * (y1 - y)), 1)
        area_b = tf.expand_dims(((x1 - x) * (y - y0)), 1)
        area_c = tf.expand_dims(((x - x0) * (y1 - y)), 1)
        area_d = tf.expand_dims(((x - x0) * (y - y0)), 1)
        output = tf.add_n([area_a*pixel_values_a,
                           area_b*pixel_values_b,
                           area_c*pixel_values_c,
                           area_d*pixel_values_d])
        return output

    def _meshgrid(self, height, width):
        x_linspace = tf.linspace(-1., 1., width)
        y_linspace = tf.linspace(-1., 1., height)
        x_coordinates, y_coordinates = tf.meshgrid(x_linspace, y_linspace)
        x_coordinates = tf.reshape(x_coordinates, [-1])
        y_coordinates = tf.reshape(y_coordinates, [-1])
        ones = tf.ones_like(x_coordinates)
        indices_grid = tf.concat([x_coordinates, y_coordinates, ones], 0)
        return indices_grid

    def _transform(self, affine_transformation, input_shape, output_size):
        batch_size = tf.shape(input_shape)[0]
        height = tf.shape(input_shape)[1]
        width = tf.shape(input_shape)[2]
        num_channels = tf.shape(input_shape)[3]

        affine_transformation = tf.reshape(affine_transformation, shape=(batch_size,2,3))

        affine_transformation = tf.reshape(affine_transformation, (-1, 2, 3))
        affine_transformation = tf.cast(affine_transformation, 'float32')

        width = tf.cast(width, dtype='float32')
        height = tf.cast(height, dtype='float32')
        output_height = output_size[0]
        output_width = output_size[1]
        indices_grid = self._meshgrid(output_height, output_width)
        indices_grid = tf.expand_dims(indices_grid, 0)
        indices_grid = tf.reshape(indices_grid, [-1]) # flatten?

        indices_grid = tf.tile(indices_grid, tf.stack([batch_size]))
        indices_grid = tf.reshape(indices_grid, (batch_size, 3, -1))

        transformed_grid = tf.matmul(affine_transformation, indices_grid)
        x_s = tf.slice(transformed_grid, [0, 0, 0], [-1, 1, -1])
        y_s = tf.slice(transformed_grid, [0, 1, 0], [-1, 1, -1])
        x_s_flatten = tf.reshape(x_s, [-1])
        y_s_flatten = tf.reshape(y_s, [-1])

        transformed_image = self._interpolate(input_shape,
                                                x_s_flatten,
                                                y_s_flatten,
                                                output_size)

        transformed_image = tf.reshape(transformed_image, shape=(batch_size,
                                                                output_height,
                                                                output_width,
                                                                num_channels))
        return transformed_image




# Group Convolution Keras Layer
# Base implementation from https://github.com/tscohen/GrouPy
#
# To use, clone https://github.com/tscohen/GrouPy and run
# python setup.py install
# (inside your virtualenv if you are using one)
#
from groupy.gconv.tensorflow_gconv.splitgconv2d import gconv2d, gconv2d_util

from keras.engine.topology import Layer

class GConv2D(Layer):
    channel_multipliers = {'Z2': 1, 'C4': 4, 'D4': 8}

    def __init__(self, out_channels=64, h_input='Z2', h_output='D4', ksize=3,strides=[1,1],
                 kernel_initializer="uniform",use_bias=True,**kwargs):
        self.out_channels_gconv = out_channels
        self.h_input = h_input
        self.h_output = h_output
        self.ksize = ksize
        self.strides=strides
        self.kwargs = kwargs
        self.kernel_initializer=kernel_initializer
        self.use_bias=use_bias
        assert ksize % 2 == 1
        assert h_input in ['Z2', 'D4', 'C4']
        assert h_output in ['D4', 'C4']
        super(GConv2D, self).__init__(**kwargs)

    def get_in_channels_gconv(self, in_channels_keras):
        input_channel_multiplier = self.channel_multipliers[self.h_input]
        assert in_channels_keras % input_channel_multiplier == 0
        in_channels_gconv = in_channels_keras // input_channel_multiplier
        return in_channels_gconv

    def get_out_channels_keras(self, out_channels_gconv):
        output_channel_multiplier = self.channel_multipliers[self.h_output]
        return out_channels_gconv * output_channel_multiplier

    def build(self, input_shape):
        in_channels_gconv = self.get_in_channels_gconv(input_shape[3]) # NHWC
        # calculate number of actual channel


        self.gconv_indices, self.gconv_shape_info, w_shape = gconv2d_util(
            h_input=self.h_input, h_output=self.h_output, in_channels=in_channels_gconv,
            out_channels=self.out_channels_gconv, ksize=self.ksize)

        # Create a trainable weight variable for this layer.
        self.w = self.add_weight(name='kernel',
                                 shape=w_shape,
                                 initializer=self.kernel_initializer,
                                 trainable=True)

        if self.use_bias:
            #bias_shape = (self.out_channels_gconv,)
            bias_shape = (self.get_out_channels_keras(self.out_channels_gconv),)

            self.b=self.add_weight(name="bias",shape=bias_shape,initializer=keras.initializers.zeros(),
                                 trainable=True)


        super(GConv2D, self).build(input_shape)  # Be sure to call this at the end



    def call(self, x):
        strides=[1,*self.strides,1] # NHWC
        output= gconv2d(input=x, filter=self.w, strides=strides, padding='SAME',
                       gconv_indices=self.gconv_indices, gconv_shape_info=self.gconv_shape_info)
        if self.use_bias:
            multiplier=self.channel_multipliers[self.h_output]
            #repeated_b=tf_repeat(self.b,[multiplier])
            #repeated_b=K.repeat_elements(self.b,multiplier,0)
            #output=K.bias_add(output,repeated_b)
            output = K.bias_add(output, self.b)

            # group_outputs=[]
            # for i in range(self.out_channels_gconv):
            #     start=i*multiplier
            #     end=i*multiplier+multiplier
            #     def group_add(v):
            #         # print(v.shape)
            #         # print(v[:,:,:,start:end].shape)
            #         # print(self.b[i])
            #         return v[:,:,:,start:end]+self.b[i]
            #     group_output=keras.layers.Lambda(group_add)(output)
            #     group_outputs.append(group_output)
            # # print(group_outputs)
            # output=keras.layers.Concatenate(axis=-1)(group_outputs)
        return output

    def compute_output_shape(self, input_shape):
        out_channels_keras = self.get_out_channels_keras(self.out_channels_gconv)
        # NHWC
        H=input_shape[1]//self.strides[0]
        W=input_shape[2]//self.strides[1]
        return input_shape[0],H,W, out_channels_keras


def tf_repeat(tensor, repeats):
    """
    Args:

    input: A Tensor. 1-D or higher.
    repeats: A list. Number of repeat for each dimension, length must be the same as the number of dimensions in input

    Returns:

    A Tensor. Has the same type as input. Has the shape of tensor.shape * repeats
    """
    with tf.variable_scope("repeat"):
        expanded_tensor = tf.expand_dims(tensor, -1)
        multiples = [1] + repeats
        tiled_tensor = tf.tile(expanded_tensor, multiples=multiples)
        repeated_tensor = tf.reshape(tiled_tensor, tf.shape(tensor) * repeats)
    return repeated_tensor