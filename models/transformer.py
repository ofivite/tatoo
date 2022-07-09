from omegaconf import OmegaConf, DictConfig
import tensorflow as tf
from tensorflow.keras.layers import Dense, MultiHeadAttention, LayerNormalization, Dropout, Softmax
from models.embedding import FeatureEmbedding

class MaskedMultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, dim_head_key, dim_head_value, dim_out):
        super(MaskedMultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.dim_head_key = dim_head_key
        self.dim_head_value = dim_head_value
        self.dim_out = dim_out
        self.att_logit_norm = tf.sqrt(tf.cast(dim_head_key, tf.float32))

        # head projection layers, separate dim  
        self.wq = tf.keras.layers.Dense(self.dim_head_key*self.num_heads) 
        self.wk = tf.keras.layers.Dense(self.dim_head_key*self.num_heads)
        self.wv = tf.keras.layers.Dense(self.dim_head_value*self.num_heads)

        self.dense = tf.keras.layers.Dense(dim_out)

    def split_heads(self, x, batch_size, head_dim):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, head_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3]) # (batch_size, num_heads, seq_len, depth)

    def masked_attention(self, q, k, v, mask=None):
        # q * k^T
        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

        # scale matmul_qk
        scaled_attention_logits = matmul_qk / self.att_logit_norm
        
        # # masked logits -> softmax 
        # scaled_attention_logits -= tf.math.reduce_max(scaled_attention_logits, axis=-1, keepdims=True) & subtract max value (to prevent nans after softmax)
        # inputs_exp = tf.exp(scaled_attention_logits)
        # if mask is not None:
        #     inputs_exp *= mask
        # inputs_sum = tf.reduce_sum(inputs_exp, axis=-1, keepdims=True)
        # attention_weights = tf.where(tf.math.not_equal(inputs_sum, 0), inputs_exp/inputs_sum, 0) 

        # add the mask to the scaled tensor.
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        # softmax normalized on the last axis (seq_len_k)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

        # if mask is not None:
        #     attention_weights *= tf.cast(~tf.cast(mask, tf.bool), tf.float32)

        # att * v
        output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

        return output, attention_weights

    def call(self, query, key, value, attention_mask, return_attention_scores=False):
        batch_size = tf.shape(query)[0]

        # project to heads space
        # -> (batch_size, seq_len, dim_head_(k/v) * num_heads)
        query = self.wq(query)  
        key = self.wk(key)
        value = self.wv(value)

        # split away head dimension and transpose
        # -> (batch_size, num_heads, seq_len_(q,k,v), dim_head_(k,v))
        query = self.split_heads(query, batch_size, self.dim_head_key)
        key = self.split_heads(key, batch_size, self.dim_head_key)
        value = self.split_heads(value, batch_size, self.dim_head_value)

        ## compute per-head attention 
        scaled_attention, attention_weights = self.masked_attention(query, key, value, attention_mask) # att=(batch_size, num_heads, seq_len_q, dim_head_value)
        
        # combine heads together
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, dim_head_value)
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.dim_head_value * self.num_heads))

        # project onto dim_out
        output = self.dense(concat_attention)  # (batch_size, seq_len_q, dim_out)

        if return_attention_scores:
            return output, attention_weights
        return output

        
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, use_masked_mha, num_heads, dim_model, dim_head_key, dim_head_value, dim_ff, dropout_rate):
        super(EncoderLayer, self).__init__()

        if use_masked_mha:
            self.mha = MaskedMultiHeadAttention(num_heads=num_heads, dim_head_key=dim_head_key, dim_head_value=dim_head_value, dim_out=dim_model)
        else:
            self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=dim_head_key, value_dim=dim_head_value, dropout=0)

        self.ffn = tf.keras.Sequential([
                          Dense(dim_ff, activation='relu'),  
                          Dense(dim_model)
                         ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def call(self, x, mask, training): # , return_attention_scores=False
        attn_output = self.mha(query=x, value=x, key=x, attention_mask=mask, return_attention_scores=False) 
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  

        ffn_output = self.ffn(out1) 
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2
    
class Encoder(tf.keras.layers.Layer):
    def __init__(self, feature_name_to_idx, embedding_kwargs, use_masked_mha, num_layers, num_heads, 
                        dim_model, dim_head_key, dim_head_value, dim_ff, dropout_rate):
        super(Encoder, self).__init__()

        self.num_layers = num_layers
        self.enc_layers = [EncoderLayer(use_masked_mha=use_masked_mha, num_heads=num_heads, dim_model=dim_model, 
                                        dim_head_key=dim_head_key, dim_head_value=dim_head_value, dim_ff=dim_ff, dropout_rate=dropout_rate) 
                                            for _ in range(self.num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

        if isinstance(embedding_kwargs, DictConfig):
            embedding_kwargs = OmegaConf.to_object(embedding_kwargs)
        # extract index of cat. features
        cat_emb_feature = embedding_kwargs.pop('cat_emb_feature')
        embedding_kwargs['cat_emb_feature_idx'] = feature_name_to_idx[cat_emb_feature]
        
        # drop specified features and extract their indices 
        features_to_drop = embedding_kwargs.pop('features_to_drop')
        embedding_kwargs['feature_idx_to_select'] = [i for f, i in feature_name_to_idx.items() if f not in features_to_drop and f != cat_emb_feature]

        self.feature_embedding = FeatureEmbedding(**embedding_kwargs)

    def call(self, x, mask, training):
        x = self.feature_embedding(x)
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, mask, training)
    
        return x

class Transformer(tf.keras.Model):
    def __init__(self, feature_name_to_idx, encoder_kwargs, decoder_kwargs):
        super().__init__()
        self.use_masked_mha = encoder_kwargs["use_masked_mha"]
        self.encoder = Encoder(feature_name_to_idx, **encoder_kwargs)
        self.dense_1 = Dense(decoder_kwargs['dim_ff_outputs'], activation='relu')
        self.dense_2 = Dense(decoder_kwargs['dim_ff_outputs']//2, activation='relu')
        self.output_dense = Dense(decoder_kwargs['n_outputs'], activation=None)
        self.output_pred = Softmax()

    def call(self, inputs, training):
        # pad ragged array to max constituent number and return normal tensor
        x = inputs.to_tensor()

        # create mask for padded tokens
        mask = self.create_padding_mask(x)
        enc_padding_mask = tf.math.logical_and(mask[:, tf.newaxis, :], mask[:, :, tf.newaxis]) # [batch, seq, seq], symmetric block-diagonal
        if self.use_masked_mha: # invert mask, 0 -> constituent, 1 -> padding
            enc_padding_mask = ~enc_padding_mask
        enc_padding_mask = tf.cast(enc_padding_mask, tf.float32)
        enc_padding_mask = enc_padding_mask[:, tf.newaxis, :, :] # additional axis for head dimension 

        # propagate through encoder
        enc_output = self.encoder(x, enc_padding_mask, training)

         # mask padded tokens before pooling 
        mask = tf.cast(mask, tf.float32)
        enc_output *= mask[...,  tf.newaxis]
        
        # pooling by summing over constituent dimension
        enc_output = tf.math.reduce_sum(enc_output, axis=1) 

        # decoder
        output = self.dense_1(enc_output)
        output = self.dense_2(output)
        output = self.output_pred(self.output_dense(output))

        return output

    def create_padding_mask(self, seq):
        mask = tf.math.reduce_any(tf.math.not_equal(seq, 0), axis=-1) # [batch, seq], 0 -> padding, 1 -> constituent
        return mask