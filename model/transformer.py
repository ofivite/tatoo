import tensorflow as tf
from tensorflow.keras.layers import Dense, MultiHeadAttention, LayerNormalization, Dropout, Softmax

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, num_heads, dim_model, dim_ff, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=dim_model, attention_axes=1)
        self.ffn = tf.keras.Sequential([
                          Dense(dim_ff, activation='relu'),  
                          Dense(dim_model)
                         ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def call(self, inputs, mask, training): # , return_attention_scores=False
        attn_output = self.mha(query=inputs, value=inputs, key=inputs, attention_mask=mask, return_attention_scores=False) 
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)  

        ffn_output = self.ffn(out1) 
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2
    
class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, dim_model, num_heads, dim_ff, dropout_rate=0.1):
        super(Encoder, self).__init__()

        self.num_layers = num_layers
        self.enc_layers = [EncoderLayer(num_heads=num_heads, dim_model=dim_model, dim_ff=dim_ff, dropout_rate=dropout_rate) 
                                            for _ in range(self.num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, mask, training):
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, mask, training)
    
        return x

class Transformer(tf.keras.Model):
    def __init__(self, num_layers, dim_model, num_heads, dim_ff, n_outputs, dropout_rate=0.1):
        super().__init__()
        self.encoder = Encoder(num_layers=num_layers, dim_model=dim_model, num_heads=num_heads, dim_ff=dim_ff, dropout_rate=dropout_rate)
        self.output_dense = Dense(n_outputs, activation=None)
        self.output_pred = Softmax()

    def call(self, inputs, training):
        x = inputs.to_tensor() # pad ragged array to max constituent number and return normal tensor
        
        # enc_padding_mask = self.create_padding_mask(x)
        enc_padding_mask = None
        enc_output = self.encoder(x, enc_padding_mask, training) 

        enc_output = tf.math.reduce_sum(enc_output, axis=1) # pooling by summing over constituent dimension
        output = self.output_pred(self.output_dense(enc_output))

        return output

    def create_padding_mask(self, seq):
        mask = tf.math.reduce_all(tf.math.equal(seq, 0), axis=2)
        mask = tf.cast(mask, dtype=tf.int32)
        # mask = tf.tensordot(single_mask, single_mask, axes=0)
        
        return mask