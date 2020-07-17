import tensorflow as tf
from data_loader import ImgLoader

class MultiTaskCNN(tf.keras.Model):

    def __init__(self, batch_size):     # you should define your layers in __init__
        super(MultiTaskCNN, self).__init__()
        self.batch_size = batch_size
        self.Conv_1 = tf.keras.layers.Conv2D(32, 7, padding='same', activation='relu') # filters, kernel_size, strides, padding
        self.MaxPool_1 = tf.keras.layers.MaxPooling2D(2) #pool_size, strides, padding, data_format
        self.Conv_2 = tf.keras.layers.Conv2D(32, 7, padding='same', activation='relu')
        self.MaxPool_2 = tf.keras.layers.MaxPooling2D(2)
        self.Conv_3 = tf.keras.layers.Conv2D(64, 5, padding = 'same', activation = 'relu')
        self.Flat = tf.keras.layers.Flatten();
        self.full_Connect_1 = tf.keras.layers.Dense(1000, activation='relu', bias_initializer='zeros')
        self.full_Connect_2 = tf.keras.layers.Dense(400, activation='relu', bias_initializer='zeros')
        self.full_Connect_3 = tf.keras.layers.Dense(324, activation='relu', bias_initializer='zeros')
        self.full_Connect_4 = tf.keras.layers.Dense(1, activation='relu', bias_initializer='zeros')


    def call(self, inputs, training=False):  # you should implement the model's forward pass in call

        x = self.Conv_1(inputs);
        x = self.MaxPool_1(x);
        x = self.Conv_2(x);
        x = self.MaxPool_2(x);
        x = self.Conv_3(x);
        x = self.Flat(x);
        x = self.full_Connect_1(x);
        x = self.full_Connect_2(x);
        x1 = x;
        density_map = tf.reshape(self.full_Connect_3(x1),[self.batch_size,18,18,1]);
        count = self.full_Connect_4(x);

        return count, density_map



