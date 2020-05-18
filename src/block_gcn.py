# -*- coding: utf-8 -*-
import dgl.function as fn
# os.environ['DGLBACKEND'] = 'mxnet'
from mxnet import gluon, nd
from mxnet.gluon import nn
from src import glc


class PreGCNSpatialAttentionLayer(gluon.HybridBlock):
    def __init__(self, m, k=64, ):  # output.shape (k,m)
        super(PreGCNSpatialAttentionLayer, self).__init__()
        self.m = m
        self.k = k
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            self.features.add(
                nn.Dense(m * k, use_bias=False),
                nn.BatchNorm(),
                nn.Activation('relu'),
            )

    def forward(self, feat):
        # B,C, H, W -> B, k, M
        B, C, H, W = feat.shape
        feat = nd.mean(feat, axis=1)  # (B,C,H,W)=> (B,H,W)
        feat = feat.reshape((B, -1))  # (B,H,W)=> (B, HW)
        feat = self.features(feat)  # (B, k*m)
        feat = feat.reshape((B, self.k, self.m))  # B,k,m
        return feat  # (B,k,m)


class GCNLayer(gluon.HybridBlock):
    '''
    A simple GCN layer.
    X' = \sigma( D^{-1} AXW )
    '''

    def __init__(self, in_feats, out_feats, activation=None, bias=False):
        super(GCNLayer, self).__init__()
        self.dense = gluon.nn.Dense(out_feats, use_bias=bias)
        self.out_feats = out_feats
        if activation is not None:
            self.activation = nn.Activation(activation, prefix=activation + '_')
        else:
            self.activation = None

    def forward(self, g, h):
        # h.shape = (M, B, d); B is batchsize
        (M, B, d0) = h.shape
        h = h.reshape((M * B, d0))
        h = self.dense(h)  # XW, (MB,d)
        h = h.reshape((M, B, -1))
        g.ndata['h'] = h
        g.update_all(fn.copy_src(src='h', out='m'),  # AXW
                     fn.sum(msg='m', out='h'))

        h = g.ndata.pop('h')
        h = h * g.ndata['D_norm']  # D^{-1}AXW
        if self.activation is not None:
            h = self.activation(h)
        h_out = h
        return h_out  # (M,B,d)


class GCN(gluon.HybridBlock):
    def __init__(self,
                 g,
                 in_feats,
                 n_hiddens,
                 n_semantic,
                 n_gcnout,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.g = g
        # self.inp_layer = gluon.nn.Dense(n_hidden, activation)
        self.layers = gluon.nn.HybridSequential()
        if glc.use_pre_sgv_out and glc.gcn_num_count > 0:
            n_hiddens = [in_feats + glc.word2vec_size, ] + n_hiddens
        else:
            glc.gcn_num_count += 1
            n_hiddens = [in_feats, ] + n_hiddens
        for i in range(len(n_hiddens) - 1):
            self.layers.add(GCNLayer(n_hiddens[i], n_hiddens[i + 1], activation, bias=True))

        self.semantic_layer = GCNLayer(n_hiddens[-1], n_semantic, activation)  # , True)
        self.out_layer = GCNLayer(n_semantic, glc.word2vec_size)
        self.dropout = gluon.nn.Dropout(rate=dropout)

    def get_semantic(self, h):  # output: nodes' feature
        for i, layer in enumerate(self.layers):
            h = layer(self.g, h)
            h = self.dropout(h)
        h = self.semantic_layer(self.g, h)  # (M, B, C)
        return h

    def get_gcn_out(self, h):  # after function get_semantic
        h = self.dropout(h)
        h = self.out_layer(self.g, h)  # (M, B, v)
        return h

    def forward(self, h):
        # this.forward() is equal : this.get_semantic() -> this.get_gcn_out()
        # h   (M,B,C)
        h = self.get_semantic(h)
        h = self.dropout(h)
        h = self.out_layer(self.g, h)  # (M, B, v)
        return h  # (M, B, v)


class SGVAttentionBlock(object):
    def __init__(self):
        pass

    def get_semantic(self, features):
        pass

    def get_sgv_attention(self, features):
        pass


class SGVSpatialAttentionBlock(gluon.HybridBlock, SGVAttentionBlock):
    def __init__(self, g, m, in_feats, n_hiddens, n_semantic, n_gcnout,
                 activation=None, bias=False, dropout=0.):
        super(SGVSpatialAttentionBlock, self).__init__()
        self.H = int(n_semantic ** (1 / 2))
        with self.name_scope():
            self.pre_gcn = PreGCNSpatialAttentionLayer(m=m, k=in_feats)
            self.gcn = GCN(g, in_feats, n_hiddens, n_semantic, n_gcnout, activation, dropout)

    def get_semantic(self, features, block_id):  # For CNN (backbone)
        # features (B, C, H, W)
        features = self.pre_gcn(features)  # (B,in_feats,m)
        features = features.transpose((2, 0, 1))  # (M,B,in_feats)
        if glc.use_pre_sgv_out and len(glc.pre_sgv_out) > 0:
            # features = features + glc.pre_sgv_out  # (M, B, in_feats)
            features = nd.concat(features, glc.pre_sgv_out, dim=-1)  # (M, B, in_feats+glc.word2vec_size)
        semantic_features = self.gcn.get_semantic(features)  # (M, B, n_semantic)

        # save all the sgv_output,
        if glc.use_pre_sgv_out or glc.use_gcn_semantic2FC:
            gcn_out = self.gcn.get_gcn_out(semantic_features)  # (M, B, n_gcnout)
            if glc.use_pre_sgv_out:
                if len(glc.pre_sgv_out) > 0:
                    glc.pre_sgv_out = glc.pre_sgv_out + gcn_out  # (M, B, n_gcnout)
                else:
                    glc.pre_sgv_out = gcn_out  # for word2vecloss
            if glc.use_gcn_semantic2FC:
                glc.gcn_semantic2FC[block_id] = gcn_out.mean(axis=0).transpose((1, 0))  # (n_gcout,B) from (B, n_gcout)

        return semantic_features  # (M, B, n_semantic)

    def get_sgv_attention(self, semantic_features):
        '''
        This attention values are for CNN(backbone)
        :param features: (B, C, H, W)
        :return: (B, C)
        use this.get_semantic before using this.get_sgv_attention
        '''
        # (B, n_semantic), i.e. (B, HW)
        spatial_attention = semantic_features.mean(axis=0)  # squeeze function
        spatial_attention = spatial_attention.reshape(-1, 1, self.H, self.H)  # (B,1,H,W)
        spatial_attention = nd.sigmoid(spatial_attention)
        return spatial_attention  # (B, 1, H, W)

    def forward(self, features):  #
        # features (B, C, H, W)
        features = self.pre_gcn(features)  # (B,C,m)
        features = features.transpose((2, 0, 1))  # (M,B,C)
        features = self.gcn(features)  # (M, B, n_gout)
        return features
