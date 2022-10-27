import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer


def conv(input_tensor, name, kw, kh, n_out, dw=1, dh=1, activation_fn=tf.nn.relu, trainable=False):
    n_in = input_tensor.get_shape()[-1].value
    with tf.variable_scope(name):
        weights = tf.get_variable('weights', [kh, kw, n_in, n_out], tf.float32, xavier_initializer(), trainable=trainable)
        biases = tf.get_variable("bias", [n_out], tf.float32, tf.constant_initializer(0.0), trainable=trainable)
        conv = tf.nn.conv2d(input_tensor, weights, (1, dh, dw, 1), padding='SAME')
        results = tf.nn.bias_add(conv, biases)
        if activation_fn is not None:
            results = activation_fn(results)
        return results


def pool(input_tensor, name, kh, kw, dh, dw):
    return tf.nn.max_pool(input_tensor,
                          ksize=[1, kh, kw, 1],
                          strides=[1, dh, dw, 1],
                          padding='VALID',
                          name=name)


class Vgg16:
    def __init__(self, scope="Vgg16", trainable=False):
        self.scope = scope
        self.trainable = trainable

    def __call__(self, inputs):
        out = list()
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            # block 1
            x = conv(inputs, name="conv1_1", kh=3, kw=3, n_out=64, trainable=self.trainable)
            out.append(x)
            x = conv(x, name="conv1_2", kh=3, kw=3, n_out=64, trainable=self.trainable)
            out.append(x)
            x = pool(x, name="pool1", kh=2, kw=2, dw=2, dh=2)

            # block 2
            x = conv(x, name="conv2_1", kh=3, kw=3, n_out=128, trainable=self.trainable)
            out.append(x)
            x = conv(x, name="conv2_2", kh=3, kw=3, n_out=128, trainable=self.trainable)
            out.append(x)
            x = pool(x, name="pool2", kh=2, kw=2, dh=2, dw=2)

            # # block 3
            x = conv(x, name="conv3_1", kh=3, kw=3, n_out=256, trainable=self.trainable)
            out.append(x)
            x = conv(x, name="conv3_2", kh=3, kw=3, n_out=256, trainable=self.trainable)
            out.append(x)
            x = conv(x, name="conv3_3", kh=3, kw=3, n_out=256, trainable=self.trainable)
            out.append(x)
            x = pool(x, name="pool3", kh=2, kw=2, dh=2, dw=2)

            # block 4
            x = conv(x, name="conv4_1", kh=3, kw=3, n_out=512, trainable=self.trainable)
            out.append(x)
            x = conv(x, name="conv4_2", kh=3, kw=3, n_out=512, trainable=self.trainable)
            out.append(x)
            x = conv(x, name="conv4_3", kh=3, kw=3, n_out=512, trainable=self.trainable)
            out.append(x)
            x = pool(x, name="pool4", kh=2, kw=2, dh=2, dw=2)

            # block 5
            x = conv(x, name="conv5_1", kh=3, kw=3, n_out=512, trainable=self.trainable)
            out.append(x)
            x = conv(x, name="conv5_2", kh=3, kw=3, n_out=512, trainable=self.trainable)
            out.append(x)
            x = conv(x, name="conv5_3", kh=3, kw=3, n_out=512, trainable=self.trainable)
            out.append(x)

        return out
