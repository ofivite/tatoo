import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Add, Softmax

class RadialFrequencies(tf.keras.Model):
    def __init__(self, hidden_dim, n_freqs):
        super().__init__()
        self.dense_1 = Dense(hidden_dim, activation=tf.nn.relu)
        self.dense_2 = Dense(hidden_dim, activation=tf.nn.relu)
        self.r_real = Dense(n_freqs+1, activation=tf.nn.relu, bias_initializer=tf.keras.initializers.ones) 
        self.r_imag = Dense(n_freqs, activation=tf.nn.relu, bias_initializer=tf.keras.initializers.ones)  
        
    def call(self, inputs):
        x_hidden = self.dense_1(inputs)
        x_hidden = self.dense_2(x_hidden)
        r_real = self.r_real(x_hidden)
        r_imag = self.r_imag(x_hidden)
        return r_real, r_imag

class WaveformEncoder(tf.keras.Model):
    def __init__(self, feature_name_to_idx, hidden_dim=16, n_freqs=4, n_filters=10, n_rotations=32):
        super().__init__()
        self.feature_name_to_idx = feature_name_to_idx
        self.n_freqs = n_freqs
        self.n_filters = n_filters
        self.n_rotations = n_rotations
        self.radial_models = [RadialFrequencies(hidden_dim, n_freqs) for _ in range(self.n_filters)]
    
    @staticmethod    
    def to_complex(real, imag):
        real = tf.concat([tf.reverse(real[..., 1:], axis=[-1]), real], axis=-1)
        zeros = tf.expand_dims(tf.zeros_like(imag[..., 0]), axis=-1)
        imag = tf.concat([-tf.reverse(imag, axis=[-1]), zeros, imag], axis=-1)
        z = tf.dtypes.complex(real, imag)
        return z

    def get_radial_spectrum(self, inputs):
        r = tf.expand_dims(inputs[..., self.feature_name_to_idx['r']], axis=-1)
        r_freqs = []
        for radial_model in self.radial_models:
            r_freqs_real, r_freqs_imag = radial_model(r)
            r_freqs.append(self.to_complex(r_freqs_real, r_freqs_imag))
        r_freqs = tf.stack(r_freqs, axis=-2)
        return r_freqs

    def get_azim_spectrum(self, inputs):
        theta = inputs[..., self.feature_name_to_idx['theta']]
        azim_freqs = [tf.math.exp(tf.dtypes.complex(0, m*theta)) for m in range(-self.n_freqs, self.n_freqs+1)]
        azim_freqs = tf.stack(azim_freqs, axis=-1)[..., tf.newaxis, :] # additional axis for filter dimension
        return azim_freqs

    def get_rotation_spectrum(self):
        rotations = tf.constant(2*np.pi/self.n_rotations, dtype=tf.float32)*tf.range(0, self.n_rotations, dtype=tf.float32)
        rotation_freqs = tf.tensordot(tf.range(-self.n_freqs, self.n_freqs+1, dtype=tf.float32), rotations, axes=0) # tensor product with output dim [2*n_freqs+1, n_rotations]
        rotation_freqs = tf.math.exp(tf.dtypes.complex(tf.constant(0., dtype=tf.float32), rotation_freqs))
        return rotation_freqs

    def sample_waveforms(self, proj_freqs):  
        rotation_freqs = self.get_rotation_spectrum()
        waveforms = tf.tensordot(proj_freqs, rotation_freqs, axes=[[2], [0]]) # axes 2 and 0 are m dimension (filter frequency)
        # if not tf.math.reduce_all((imag_part:=tf.math.abs(tf.math.imag(waveforms))) < 1.e-5):
        #     print(waveforms[imag_part > 1.e-5])
        #     raise RuntimeError('Found large elements in imaginary part of waveforms')
        waveforms = tf.math.abs(waveforms)
        return waveforms

    def project_onto_filters(self, inputs, filter_freqs):
        z = inputs[..., self.feature_name_to_idx['pt']]
        z = tf.dtypes.complex(z, 0)[..., tf.newaxis, tf.newaxis] # axes for filter and frequency dimensions
        proj_freqs = tf.math.reduce_sum(tf.multiply(z, filter_freqs), axis=1) # sum over constituents
        # assert tf.math.reduce_all(tf.math.imag(proj_freqs + tf.reverse(proj_freqs, axis=[-1])) == 0)
        return proj_freqs

    def call(self, inputs):
        r_freqs = self.get_radial_spectrum(inputs)
        azim_freqs = self.get_azim_spectrum(inputs)
        filter_freqs = tf.math.multiply(r_freqs, azim_freqs)
        proj_freqs = self.project_onto_filters(inputs, filter_freqs)
        waveforms = self.sample_waveforms(proj_freqs)
        return waveforms

class WaveformDecoder(tf.keras.Model):
    def __init__(self, kernel_size=3, n_kernels=10, hidden_dim=10, n_outputs=2):
        super().__init__()
        self.conv_1 = Conv1D(n_kernels, kernel_size, padding='same', data_format='channels_last', activation='relu')
        self.conv_2 = Conv1D(n_kernels, kernel_size, padding='same', data_format='channels_last', activation='relu')
        # self.conv_3 = Conv1D(1, kernel_size, padding='same', data_format='channels_last', activation='relu')
        self.add = Add()
        self.flatten = Flatten() 
        self.dense_1 = Dense(hidden_dim, activation=tf.nn.relu)
        self.dense_2 = Dense(hidden_dim//2, activation=tf.nn.relu)
        self.output_dense = Dense(n_outputs, activation=None)
        self.output_pred = Softmax()
        
    def call(self, inputs):
        x_conv_out = self.conv_1(inputs)
        x_conv_in = self.add([x_conv_out, inputs])
        x_conv_out = self.conv_2(x_conv_in)
        x_conv_in = self.add([x_conv_out, inputs])
        # x_conv = self.conv_3(x_conv)
        # x_conv = tf.squeeze(x_conv, axis=-1)
        x_conv_out = self.flatten(x_conv_in)
        x_dense = self.dense_1(x_conv_out)
        x_dense = self.dense_2(x_dense)
        outputs = self.output_pred(self.output_dense(x_dense))
        return outputs

class TacoNet(tf.keras.Model):
    def __init__(self, feature_name_to_idx, hidden_dim_encoder=16, n_freqs=4, n_filters=10, n_rotations=32, 
                    kernel_size=3, n_kernels=10, hidden_dim_decoder=10, n_outputs=2):
        super().__init__()
        self.wave_encoder = WaveformEncoder(feature_name_to_idx, hidden_dim_encoder, n_freqs, n_filters, n_rotations)
        self.wave_decoder = WaveformDecoder(kernel_size, n_kernels, hidden_dim_decoder, n_outputs)
        
    def call(self, inputs):
        waveforms = self.wave_encoder(inputs)
        outputs = self.wave_decoder(waveforms[..., tf.newaxis])
        return outputs