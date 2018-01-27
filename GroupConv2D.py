#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 21:45:43 2018

@author: xi
"""
import keras.backend as K
from keras.engine import InputSpec
from keras.layers import Conv2D,concatenate,BatchNormalization,Activation,ZeroPadding2D,Dense,GlobalAveragePooling2D,MaxPooling2D,Input
class GroupConv2D(Conv2D):
    def __init__(self, filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 num_group=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(GroupConv2D, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)
        self.num_group = num_group
        if self.filters % self.num_group != 0:
            raise ValueError("filters must divided by num_group with no remainders!")
        self.input_spec = InputSpec(ndim=4)

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        if input_dim % self.num_group != 0:
            raise ValueError("The channel dimension of input tensor must divided by num_group with no remainders!")

        kernel_shape = self.kernel_size + (input_dim/self.num_group, self.filters)

        self.kernel = self.add_weight(kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight((self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})
        self.built = True
        self.channel_num = input_dim


    def call(self, inputs):
        filter_in_group = self.filters / self.num_group
        if self.data_format == 'channels_first':
            channel_axis = 1
            input_in_group = self.channel_num / self.num_group
            outputs_list = []
            for i in range(self.num_group):
                outputs = K.conv2d(
                    inputs[:,i*input_in_group:(i+1)*input_in_group,:,:],
                    self.kernel[:, :, :, i*filter_in_group:(i+1)*filter_in_group],
                    strides=self.strides,
                    padding=self.padding,
                    data_format=self.data_format,
                    dilation_rate=self.dilation_rate)

                if self.use_bias:
                    outputs = K.bias_add(
                                         outputs,
                                         self.bias[i*filter_in_group:(i+1)*filter_in_group],
                                         data_format=self.data_format)
                outputs_list.append(outputs)

        elif self.data_format == 'channels_last':
            outputs_list = []
            channel_axis = -1
            input_in_group = self.channel_num / self.num_group
            for i in range(self.num_group):
                outputs = K.conv2d(
                    inputs[:, :, :, i*input_in_group:(i+1)*input_in_group],
                    self.kernel[:, :, :, i*filter_in_group:(i+1)*filter_in_group],
                    strides=self.strides,
                    padding=self.padding,
                    data_format=self.data_format,
                    dilation_rate=self.dilation_rate)

                if self.use_bias:
                    outputs = K.bias_add(
                                         outputs,
                                         self.bias[i*filter_in_group:(i+1)*filter_in_group],
                                         data_format=self.data_format)
                outputs_list.append(outputs)

        outputs = concatenate(outputs_list, axis=channel_axis)
        return outputs

    def get_config(self):
        config = super(Conv2D, self).get_config()
        config.pop('rank')
        config["num_group"] = self.num_group
        return config
    
    
def unit(base_name, num_filters, input_tensor):
    x = Conv2D(num_filters[0], (1, 1), use_bias=False, name=base_name+"conv1")(input_tensor)
    x = BatchNormalization(epsilon=epsilon, momentum=0.9, axis=3, name=base_name+"bn1")(x)
    x = Activation("relu", name=base_name+"relu1")(x)

    x = ZeroPadding2D((1, 1))(x)
    x = GroupConv2D(num_filters[1], (3, 3), use_bias=False, num_group=32, name=base_name+"conv2")(x)
    x = BatchNormalization(epsilon=epsilon, momentum=0.9, axis=3, name=base_name+"bn2")(x)
    x = Activation("relu", name=base_name+"relu2")(x)

    x = Conv2D(num_filters[2], (1, 1), use_bias=False, name=base_name+"conv3")(x)
    x = BatchNormalization(epsilon=epsilon, momentum=0.9, axis=3, name=base_name+"bn3")(x)

    return x


def stage(stage_id, num_unit, num_filters, input_tensor):
    base_name = "stage%d_unit%d_"
    for i in range(1, num_unit+1):
        x1 = input_tensor
        x1 = unit(base_name=base_name % (stage_id, i), num_filters=num_filters, input_tensor=x1)
        if i == 1:
            x2 = Conv2D(num_filters[2], (1, 1), use_bias=False, name=base_name % (stage_id, i)+"sc")(input_tensor)
            x2 = BatchNormalization(epsilon=epsilon, momentum=0.9, name=base_name % (stage_id, i)+"sc_bn")(x2)
        else:
            x2 = input_tensor
        input_tensor = add([x1, x2])
        input_tensor = Activation("relu", name=base_name % (stage_id, i)+"relu")(input_tensor)

    return input_tensor


def ResNext50():
    #input_tensor = Input(shape=(224, 224, 3), name="input")
    input_tensor = Input(shape=(32, 32, 3), name="input")
    x = BatchNormalization(epsilon=epsilon, momentum=0.99, axis=3, name="bn_data")(input_tensor)
    x = ZeroPadding2D((3, 3))(x)
    x = Conv2D(64, (7, 7), strides=(2, 2), use_bias=False, name="conv0")(x)
    x = BatchNormalization(epsilon=epsilon, momentum=0.99, axis=3, name="bn0")(x)
    x = Activation("relu", name="relu0")(x)
    x = ZeroPadding2D((1, 1))(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = stage(1, 3, [128, 128, 256], x)
    x = stage(2, 4, [256, 256, 512], x)
    x = stage(3, 6, [512, 512, 1024], x)
    x = stage(4, 3, [1024, 1024, 2048], x)

    x = GlobalAveragePooling2D(name="pool1")(x)
    #x = Dense(1000, name="fc1", activation="softmax")(x)
    x = Dense(10, name="fc1", activation="softmax")(x)
    model = Model(input_tensor, x)
    return model