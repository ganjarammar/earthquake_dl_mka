import tensorflow as tf
# from .attention import MultiHeadAttention
from tensorflow.keras.layers import MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    def __init__(self, conv_units=32, conv_size=3, pool_units=2, suffix=None, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.conv_units = conv_units
        self.conv_size  = conv_size
        self.pool_units = pool_units
        
        self.conv  = tf.keras.layers.SeparableConv1D(
            conv_units, (conv_size,), padding='same', activation=None,
            name='conv{}'.format(suffix)
        )
        self.maxp  = tf.keras.layers.MaxPooling1D(pool_units, name='maxp{}'.format(suffix))
        self.bnorm = tf.keras.layers.BatchNormalization(name='bnorm{}'.format(suffix))
        self.act   = tf.keras.layers.Activation('relu', name='relu{}'.format(suffix))

    
    def get_config(self):
        config = super(EncoderBlock, self).get_config()
        config.update({
            'conv_units': self.conv_units,
            'conv_size': self.conv_size,
            'pool_units': self.pool_units
        })
        return config


    def call(self, inputs, mask=None):
        return self.act(self.bnorm(self.maxp(self.conv(inputs))))


class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self, conv_units=32, conv_size=3, upsample_units=2, suffix=None, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.conv_units = conv_units
        self.conv_size  = conv_size
        self.upsample_units = upsample_units

        self.upsample = tf.keras.layers.UpSampling1D(upsample_units, name='upsample{}'.format(suffix))
        self.conv     = tf.keras.layers.SeparableConv1D(
            conv_units, (conv_size,), padding='same', activation='relu',
            name='conv{}'.format(suffix)
        )

    def get_config(self):
        config = super(DecoderBlock, self).get_config()
        config.update({
            'conv_units': self.conv_units,
            'conv_size': self.conv_size,
            'upsample_units': self.upsample_units
        })
        return config

    def call(self, inputs, mask=None):
        return self.conv(self.upsample(inputs))


class IndoEQ(tf.keras.models.Model):
    def __init__(self, **kwargs):
        super(IndoEQ, self).__init__(**kwargs)

        self.encoder_blocks = []
        for i in range(5):
            conv_units = int(1 << (i + 2))
            conv_size  = 11 - 2 * i
            pool_units = 2 if i < 4 else 3
            encoder_block = EncoderBlock(
                conv_units=conv_units, conv_size=conv_size, pool_units=pool_units,
                suffix=i+1, name='encoder_block_{}'.format(i+1)
            )
            self.encoder_blocks.append(encoder_block)

        self.blstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True), name='blstm')
        self.lstm1 = tf.keras.layers.LSTM(32, return_sequences=True, name='lstm1')
        self.lstm2 = tf.keras.layers.LSTM(32, return_sequences=True, name='lstm2')

        self.attn1 = MultiHeadAttention(num_heads=8, key_dim=32, value_dim=64, name='attn1')

        self.lstm_s = tf.keras.layers.LSTM(16, return_sequences=True, name='lstm_s')
        self.lstm_p = tf.keras.layers.LSTM(16, return_sequences=True, name='lstm_p')

        self.decoder_blocks_p, self.decoder_blocks_s, self.decoder_blocks_d = [], [], []
        for i in range(5):
            units = int(1 << (5 - i))
            conv_size = 2 * i + 3
            upsample_units = 3 if i == 0 else 2
            decoder_block_p = DecoderBlock(
                conv_units=units, conv_size=conv_size, upsample_units=upsample_units,
                suffix=i+1, name='decoder_block_p_{}'.format(i+1)
            )
            decoder_block_s = DecoderBlock(
                conv_units=units, conv_size=conv_size, upsample_units=upsample_units,
                suffix=i+1, name='decoder_block_s_{}'.format(i+1)
            )
            decoder_block_d = DecoderBlock(
                conv_units=units, conv_size=conv_size, upsample_units=upsample_units,
                suffix=i+1, name='decoder_block_d_{}'.format(i+1)
            )
            self.decoder_blocks_p.append(decoder_block_p)
            self.decoder_blocks_s.append(decoder_block_s)
            self.decoder_blocks_d.append(decoder_block_d)

        self.sigmoid_p = tf.keras.layers.Conv1D(1, (13,), padding='same', activation='sigmoid', name='picker_p')
        self.sigmoid_s = tf.keras.layers.Conv1D(1, (13,), padding='same', activation='sigmoid', name='picker_s')
        self.sigmoid_d = tf.keras.layers.Conv1D(1, (13,), padding='same', activation='sigmoid', name='detector')
        

    def get_config(self):
        config = super(IndoEQ, self).get_config()
        return config

    def call(self, inputs, mask=None):
        x = inputs
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
        
        x = self.lstm2(self.lstm1(self.blstm(x)))
        x, att_weights = self.attn1(query=x, value=x, key=x, return_attention_scores=True)
        return x
        # x_p = self.lstm_p(x)
        # x_s = self.lstm_s(x)

        # for decoder_block_p, decoder_block_s, decoder_block_d in zip(
        #     self.decoder_blocks_p, self.decoder_blocks_s, self.decoder_blocks_d
        # ):
        #     x_p = decoder_block_p(x_p)
        #     x_s = decoder_block_s(x_s)
        #     x = decoder_block_d(x)

        # x_s = self.sigmoid_s(x_s)
        # x_p = self.sigmoid_p(x_p)
        # x = self.sigmoid_d(x)

        # return x, x_p, x_s

    def make(self, input_shape):
        x = tf.keras.layers.Input(shape=input_shape, name='input')
        model = tf.keras.Model(inputs=[x], outputs=self.call(x), name='indoeq_model')
        return model
