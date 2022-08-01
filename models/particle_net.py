from omegaconf import OmegaConf, DictConfig
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Conv2D, Activation


class ParticleNet(tf.keras.Model):
    def __init__(self, encoder_cfg, decoder_cfg):
        super(ParticleNet, self).__init__()

        self.num_classes = decoder_cfg['output_classes']
        self.conv_params = [(layer_setup[0], tuple(layer_setup[1]),) for layer_setup in encoder_cfg['conv_params']]
        self.conv_pooling = encoder_cfg['conv_pooling']
        self.fc_params = [(layer_setup, encoder_cfg['dropout_rate'],) for layer_setup in encoder_cfg['dense_params']]
        self.num_points = encoder_cfg['seq_cutoff_len']

        # Initialising layers
        self.batch_norm = BatchNormalization()
        
        self.edge_conv_layers = []
        for layer_idx, layer_param in enumerate(self.conv_params):
            K, channels = layer_param
            self.edge_conv_layers.append(
                EdgeConv(self.num_points, K, channels, with_bn=True, activation='relu', pooling=self.conv_pooling, name=f'{self.name}_EdgeConv_{layer_idx}')
            )

        self.dense_layers = []
        self.dense_dropout = []
        if self.fc_params is not None:
            for layer_idx, layer_param in enumerate(self.fc_params):
                units, dropout_rate = layer_param

                self.dense_layers.append(
                    Dense(units, activation='relu')
                )

                if dropout_rate is not None and dropout_rate > 0:
                    self.dense_dropout.append(
                        Dropout(dropout_rate)
                    )
            
            self.out = Dense(self.num_classes, activation='softmax')

    @tf.function
    def call(self, input):
        masking = True

        features = input[0][:,:self.num_points,:].to_tensor() # (batch_size, particles, features), here: (128, 125, 22)
        # print(f'FEATURES - type: {type(features)}, shape: {features.shape}')

        # Converting from (r, theta)-space to (eta, phi)-space
        eta = features[:,:,0] * tf.math.cos(features[:,:,1]) # (128, 125)
        phi = features[:,:,0] * tf.math.sin(features[:,:,1])
        eta = eta[:, :, tf.newaxis] # (128, 125, 1)
        phi = phi[:, :, tf.newaxis]

        points = tf.concat([eta, phi], -1) # (128, 125, 2)

        if masking:
            reduced_fts = tf.expand_dims(tf.reduce_sum(features,-1),-1)  # From (128, 125, 22) to (128, 125, 1)
            mask = tf.cast(tf.not_equal(reduced_fts, 0), dtype='float32')
            coord_shift = tf.multiply(1e9, tf.cast(tf.equal(mask, 0), dtype='float32'))
        
        fts = tf.squeeze(self.batch_norm(tf.expand_dims(features, axis=2)), axis=2)

        for layer_idx, layer_param in enumerate(self.conv_params):
            if masking:
                pts = tf.add(coord_shift, points) if layer_idx == 0 else tf.add(coord_shift, fts)
            else:
                pts = points

            fts = self.edge_conv_layers[layer_idx](pts, fts)

        fts = tf.math.multiply(fts, mask) if masking else fts
        pool = tf.reduce_mean(fts, axis=1)  # (N, C)

        x = pool
        for layer_idx, layer_param in enumerate(self.fc_params):
            units, drop_rate = layer_param
            x = self.dense_layers[layer_idx](x)
            if drop_rate is not None and drop_rate > 0:
                x = self.dense_dropout[layer_idx](x)

        out = self.out(x)

        return out  # (N, num_classes)

    def create_padding_mask(self, seq):
        mask = tf.math.reduce_any(tf.math.not_equal(seq, 0), axis=-1) # [batch, seq], 0 -> padding, 1 -> constituent
        return mask


class EdgeConv(tf.keras.layers.Layer):
    def __init__(self, num_points, K, channels, with_bn=True, activation='relu', pooling='average', seq_cutoff_len=125, **kwargs):
        super(EdgeConv, self).__init__()

        self.num_points = num_points
        self.K = K
        self.channels = channels
        self.with_bn = with_bn
        self.activation = activation
        self.pooling = pooling

        self.conv2d_layers = []
        self.batchnorm_layers = []
        self.activation_layers = []

        for idx, channel in enumerate(self.channels):
            self.conv2d_layers.append(
                Conv2D(channel, kernel_size=(1,1), strides=1, data_format='channels_last', use_bias=False if self.with_bn else True, kernel_initializer='glorot_normal')
            )

            if self.with_bn:
                self.batchnorm_layers.append(BatchNormalization())
            if self.activation:
                self.activation_layers.append(Activation(self.activation))

        self.shortcut = Conv2D(self.channels[-1], kernel_size=(1,1), strides=1, data_format='channels_last', use_bias=False if self.with_bn else True, kernel_initializer='glorot_normal')
        if self.with_bn:
            self.shortcut_batchnorm = BatchNormalization()
        if self.activation:
            self.shortcut_activation = Activation(self.activation)

    def call(self, points, features):
        d = self.batch_distance_matrix_general(points, points)
        _, indicies = tf.nn.top_k(-d, k=self.K + 1)
        indicies = indicies[:,:,1:]

        fts = features
        knn_fts = self.knn(self.num_points, self.K, indicies, fts)
        knn_fts_center = tf.tile(tf.expand_dims(fts, axis=2), (1, 1, self.K, 1))  # (N, P, K, C)
        knn_fts = tf.concat([knn_fts_center, tf.subtract(knn_fts, knn_fts_center)], axis=-1)  # (N, P, K, 2*C)

        x = knn_fts
        for idx, channel in enumerate(self.channels):
            x = self.conv2d_layers[idx](x)
            if self.with_bn:
                x = self.batchnorm_layers[idx](x)
            if self.activation:
                x = self.activation_layers[idx](x)

        if self.pooling == 'max':
            fts = tf.reduce_max(x, axis=2)  # (N, P, C')
        else:
            fts = tf.reduce_mean(x, axis=2)  # (N, P, C')

        # shortcut
        sc = self.shortcut(tf.expand_dims(features, axis=2))
        if self.with_bn:
            sc = self.shortcut_batchnorm(sc)
        sc = tf.squeeze(sc, axis=2)

        if self.activation:
            return self.shortcut_activation(sc + fts)  # (N, P, C')
        else:
            return sc + fts

    def knn(self, num_points, k, topk_indices, features):
        # topk_indices: (N, P, K)
        # features: (N, P, C)
        with tf.name_scope('knn'):
            queries_shape = tf.shape(features)
            batch_size = queries_shape[0]
            batch_indices = tf.tile(tf.reshape(tf.range(batch_size), (-1, 1, 1, 1)), (1, num_points, k, 1))
            indices = tf.concat([batch_indices, tf.expand_dims(topk_indices, axis=3)], axis=3)  # (N, P, K, 2)
            return tf.gather_nd(features, indices)

    def batch_distance_matrix_general(self, A, B):
        with tf.name_scope('dmat'):
            r_A = tf.reduce_sum(A * A, axis=2, keepdims=True)
            r_B = tf.reduce_sum(B * B, axis=2, keepdims=True)
            m = tf.matmul(A, tf.transpose(B, perm=(0, 2, 1)))
            D = r_A - 2 * m + tf.transpose(r_B, perm=(0, 2, 1))
            return D