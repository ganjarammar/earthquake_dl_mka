import tensorflow as tf

class GlobalAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(GlobalAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W1 = self.add_weight(name='W1', shape=(input_shape[-1], input_shape[-1]),
                                  initializer='random_normal', trainable=True)
        self.W2 = self.add_weight(name='W2', shape=(input_shape[-1], input_shape[-1]),
                                  initializer='random_normal', trainable=True)
        self.V = self.add_weight(name='V', shape=(input_shape[-1], 1), initializer='random_normal', trainable=True)
        super(GlobalAttention, self).build(input_shape)

    def call(self, inputs):
        # inputs shape: (batch_size, seq_len, hidden_dim)
        # score shape: (batch_size, seq_len, hidden_dim)
        score = tf.nn.tanh(tf.matmul(inputs, self.W1) + tf.matmul(tf.expand_dims(tf.reduce_mean(inputs, axis=1), 1), self.W2))
        # attention_weights shape: (batch_size, seq_len, 1)
        attention_weights = tf.nn.softmax(tf.matmul(score, self.V), axis=1)
        # context_vector shape: (batch_size, hidden_dim)
        context_vector = attention_weights * inputs
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector


class LocalAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(LocalAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W1 = self.add_weight(name='W1', shape=(input_shape[-1], input_shape[-1]),
                                  initializer='random_normal', trainable=True)
        self.W2 = self.add_weight(name='W2', shape=(input_shape[-1], input_shape[-1]),
                                  initializer='random_normal', trainable=True)
        self.V = self.add_weight(name='V', shape=(input_shape[-1], 1), initializer='random_normal', trainable=True)
        super(LocalAttention, self).build(input_shape)

    def call(self, inputs):
        # inputs shape: (batch_size, seq_len, hidden_dim)
        # score shape: (batch_size, seq_len, hidden_dim)
        score = tf.nn.tanh(tf.matmul(inputs, self.W1) + tf.matmul(inputs, self.W2))
        # attention_weights shape: (batch_size, seq_len, 1)
        attention_weights = tf.nn.softmax(tf.matmul(score, self.V), axis=1)
        # context_vector shape: (batch_size, hidden_dim)
        context_vector = attention_weights * inputs
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector


class SimplifiedEQModel(tf.keras.Model):
    def __init__(self):
        super(SimplifiedEQModel, self).__init__()
        self.lstm = tf.keras.layers.LSTM(units=32, return_sequences=True, name='lstm')
        self.conv1 = tf.keras.layers.Conv1D(32, (2, 2), )
        # self.gatt = GlobalAttention(name='global_attention')
        self.lstm_s = tf.keras.layers.LSTM(units=32, return_sequences=True, name='lstm_s')
        self.lstm_p = tf.keras.layers.LSTM(units=32, return_sequences=True, name='lstm_p')
        # self.latt_s = LocalAttention(name='local_attention_s')
        # self.latt_p = LocalAttention(name='local_attention_p')
        self.dec_d = tf.keras.layers.UpSampling1D(size=2, name='upsampling_d_1')
        self.dec_s = tf.keras.layers.UpSampling1D(size=2, name='upsampling_s_1')
        self.dec_p = tf.keras.layers.UpSampling1D(size=2, name='upsampling_p_1')
        self.dec_d_conv = tf.keras.layers.Conv1D(filters=32, kernel_size=2, padding='same', activation='relu', name='dec_conv_d_1')
        self.dec_s_conv = tf.keras.layers.Conv1D(filters=32, kernel_size=2, padding='same', activation='relu', name='dec_conv_s_1')
        self.dec_p_conv = tf.keras.layers.Conv1D(filters=32, kernel_size=2, padding='same', activation='relu', name='dec_conv_p_1')
        self.final_conv = tf.keras.layers.Conv1D(filters=1, kernel_size=2, padding='same', activation='sigmoid', name='final_conv')

        # self.build(input_shape)

    def call(self, inputs):
        # inp = self.input_layer(inputs)
        x = self.lstm(inputs)
        # g = self.gatt(x)
        # s = self.lstm_s(x)
        # p = self.lstm_p(x)
        # s = self.latt_s(x)
        # p = self.latt_p(x)
        dec_d = self.dec_d(x)
        dec_s = self.dec_s(x)
        dec_p = self.dec_p(x)
        dec_d = self.dec_d_conv(dec_d)
        dec_s = self.dec_s_conv(dec_s)
        dec_p = self.dec_p_conv(dec_p)
        dec = tf.concat([dec_d, dec_s, dec_p], axis=-1)
        out = self.final_conv(dec)
        return x
