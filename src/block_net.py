# -*- coding: utf-8 -*-
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn
from mxnet.gluon.nn import BatchNorm


# Helpers
def _conv3x3(channels, stride, in_channels):
    return nn.Conv2D(channels, kernel_size=3, strides=stride, padding=1,
                     use_bias=False, in_channels=in_channels)


# Blocks
class BasicBlockV1(HybridBlock):
    def __init__(self, channels, stride, downsample=False, in_channels=0,
                 last_gamma=False, use_se=False, norm_layer=BatchNorm, norm_kwargs=None, sgv_model=None, block_id=-1,
                 **kwargs):
        super(BasicBlockV1, self).__init__(**kwargs)
        self.body = nn.HybridSequential(prefix='')
        self.body.add(_conv3x3(channels, stride, in_channels))
        self.body.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
        self.body.add(nn.Activation('relu'))
        self.body.add(_conv3x3(channels, 1, channels))
        self.act = nn.Activation('relu')
        if not last_gamma:
            self.body.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
        else:
            self.body.add(norm_layer(gamma_initializer='zeros',
                                     **({} if norm_kwargs is None else norm_kwargs)))

        if use_se:
            self.se = nn.HybridSequential(prefix='')
            self.se.add(nn.Dense(channels // 4, use_bias=False))
            self.se.add(nn.Activation('relu'))
            self.se.add(nn.Dense(channels, use_bias=False))
            self.se.add(nn.Activation('sigmoid'))
        else:
            self.se = None

        if downsample:
            self.downsample = nn.HybridSequential(prefix='')
            self.downsample.add(nn.Conv2D(channels, kernel_size=1, strides=stride,
                                          use_bias=False, in_channels=in_channels))
            self.downsample.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
        else:
            self.downsample = None

    def hybrid_forward(self, F, x):
        residual = x

        x = self.body(x)

        if self.se:
            w = F.contrib.AdaptiveAvgPooling2D(x, output_size=1)
            w = self.se(w)
            x = F.broadcast_mul(x, w.expand_dims(axis=2).expand_dims(axis=2))

        if self.downsample:
            residual = self.downsample(residual)

        x = self.act(residual + x)  # F.Activation(residual + x, act_type='relu')

        return x


class BottleneckV1(HybridBlock):
    def __init__(self, channels, stride, downsample=False, in_channels=0,
                 last_gamma=False, use_se=False, norm_layer=BatchNorm, norm_kwargs=None, sgv_model=None, block_id=-1,
                 **kwargs):
        super(BottleneckV1, self).__init__(**kwargs)
        self.body = nn.HybridSequential(prefix='')
        self.body.add(nn.Conv2D(channels // 4, kernel_size=1, strides=stride))
        self.body.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
        self.body.add(nn.Activation('relu'))
        self.body.add(_conv3x3(channels // 4, 1, channels // 4))
        self.body.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
        self.body.add(nn.Activation('relu'))
        self.body.add(nn.Conv2D(channels, kernel_size=1, strides=1))
        self.act = nn.Activation('relu')
        if use_se:
            self.se = nn.HybridSequential(prefix='')
            self.se.add(nn.Dense(channels // 4, use_bias=False))
            self.se.add(nn.Activation('relu'))
            self.se.add(nn.Dense(channels, use_bias=False))
            self.se.add(nn.Activation('sigmoid'))
        else:
            self.se = None

        if not last_gamma:
            self.body.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
        else:
            self.body.add(norm_layer(gamma_initializer='zeros',
                                     **({} if norm_kwargs is None else norm_kwargs)))

        if downsample:
            self.downsample = nn.HybridSequential(prefix='')
            self.downsample.add(nn.Conv2D(channels, kernel_size=1, strides=stride,
                                          use_bias=False, in_channels=in_channels))
            self.downsample.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
        else:
            self.downsample = None

    def hybrid_forward(self, F, x):
        residual = x

        x = self.body(x)

        if self.se:
            w = F.contrib.AdaptiveAvgPooling2D(x, output_size=1)
            w = self.se(w)
            x = F.broadcast_mul(x, w.expand_dims(axis=2).expand_dims(axis=2))

        if self.downsample:
            residual = self.downsample(residual)

        x = self.act(x + residual)
        return x


# Blocks
class SGVAttentionBasicBlockV1(HybridBlock):
    def __init__(self, channels, stride, downsample=False, in_channels=0,
                 last_gamma=False, use_se=False, norm_layer=BatchNorm, norm_kwargs=None, sgv_model=None, block_id=-1,
                 **kwargs):
        super(SGVAttentionBasicBlockV1, self).__init__(**kwargs)
        self.body = nn.HybridSequential(prefix='')
        self.body.add(_conv3x3(channels, stride, in_channels))
        self.body.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
        self.body.add(nn.Activation('relu'))
        self.body.add(_conv3x3(channels, 1, channels))
        self.act = nn.Activation('relu')
        self.sgv_model = sgv_model
        self.block_id = block_id
        print('use_sgv_model', self.sgv_model)
        if not last_gamma:
            self.body.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
        else:
            self.body.add(norm_layer(gamma_initializer='zeros',
                                     **({} if norm_kwargs is None else norm_kwargs)))
        
        if use_se:
            self.se = nn.HybridSequential(prefix='')
            self.se.add(nn.Dense(channels // 4, use_bias=False))
            self.se.add(nn.Activation('relu'))
            self.se.add(nn.Dense(channels, use_bias=False))
            self.se.add(nn.Activation('sigmoid'))
        else:
            self.se = None

        if downsample:
            self.downsample = nn.HybridSequential(prefix='')
            self.downsample.add(nn.Conv2D(channels, kernel_size=1, strides=stride,
                                          use_bias=False, in_channels=in_channels))
            self.downsample.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
        else:
            self.downsample = None

    def hybrid_forward(self, F, x):
        residual = x

        x = self.body(x)

        if self.se:
            w = F.contrib.AdaptiveAvgPooling2D(x, output_size=1)
            w = self.se(w)
            x = F.broadcast_mul(x, w.expand_dims(axis=2).expand_dims(axis=2))

        if self.downsample:
            residual = self.downsample(residual)

        if self.sgv_model: # if this block contains gcn_block
            semantic_features = self.sgv_model.get_semantic(x, self.block_id)  # (M, B, n_semantic)
            sgv_attention = self.sgv_model.get_sgv_attention(semantic_features)  # (B,1,H,W)
            x = x * sgv_attention
        x = self.act(residual + x)
        return x


class SGVAttentionBottleneckV1(HybridBlock):
    def __init__(self, channels, stride, downsample=False, in_channels=0,
                 last_gamma=False, use_se=False, norm_layer=BatchNorm, norm_kwargs=None, sgv_model=None, block_id=-1,
                 **kwargs):
        super(SGVAttentionBottleneckV1, self).__init__(**kwargs)
        self.body = nn.HybridSequential(prefix='')
        self.body.add(nn.Conv2D(channels // 4, kernel_size=1, strides=stride))
        self.body.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
        self.body.add(nn.Activation('relu'))
        self.body.add(_conv3x3(channels // 4, 1, channels // 4))
        self.body.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
        self.body.add(nn.Activation('relu'))
        self.body.add(nn.Conv2D(channels, kernel_size=1, strides=1))
        self.act = nn.Activation('relu')
        self.sgv_model = sgv_model
        self.block_id = block_id
        if use_se:
            self.se = nn.HybridSequential(prefix='')
            self.se.add(nn.Dense(channels // 4, use_bias=False))
            self.se.add(nn.Activation('relu'))
            self.se.add(nn.Dense(channels, use_bias=False))
            self.se.add(nn.Activation('sigmoid'))
        else:
            self.se = None

        if not last_gamma:
            self.body.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
        else:
            self.body.add(norm_layer(gamma_initializer='zeros',
                                     **({} if norm_kwargs is None else norm_kwargs)))

        if downsample:
            self.downsample = nn.HybridSequential(prefix='')
            self.downsample.add(nn.Conv2D(channels, kernel_size=1, strides=stride,
                                          use_bias=False, in_channels=in_channels))
            self.downsample.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
        else:
            self.downsample = None

    def hybrid_forward(self, F, x):
        residual = x

        x = self.body(x)

        if self.se:
            w = F.contrib.AdaptiveAvgPooling2D(x, output_size=1)
            w = self.se(w)
            x = F.broadcast_mul(x, w.expand_dims(axis=2).expand_dims(axis=2))

        if self.downsample:
            residual = self.downsample(residual)

        if self.sgv_model:
            semantic_features = self.sgv_model.get_semantic(x,self.block_id)  # (M, B, n_semantic)
            sgv_attention = self.sgv_model.get_sgv_attention(semantic_features)  # (B,1,H,W)
            x = x * sgv_attention

        x = self.act(x + residual)
        return x
