import tensorflow as tf
from model.mycGAN import ops


class Generator:
    def __init__(self, out_channels, ch, use_bias=False, sn=False, scope=None, training=True):
        self.out_channels = out_channels
        self.ch = ch
        self.use_bias = use_bias
        self.sn = sn
        self.scope = scope
        self.training = training

    def __call__(self, inputs):
        with tf.variable_scope(self.scope):
            skips = list()
            layers = list()
            ch = 2 * self.ch
            x = ops.resblock_down(inputs, ch, self.use_bias, self.training, self.sn, 'encoder_1')
            skips.append(x)
            layers.append(x)
            # x = ops.self_attention_2(x, ch, self.sn, 'self_attention_1')
            ch = 2 * ch
            x = ops.resblock_down(x, ch, self.use_bias, self.training, self.sn, 'encoder_2')
            skips.append(x)
            layers.append(x)
            ch = 2 * ch
            x = ops.resblock_down(x, ch, self.use_bias, self.training, self.sn, 'encoder_3')
            skips.append(x)
            layers.append(x)
            x = ops.resblock_down(x, ch, self.use_bias, self.training, self.sn, 'encoder_4')
            skips.append(x)
            layers.append(x)
            ch = 2 * ch
            x = ops.resblock_down(x, ch, self.use_bias, self.training, self.sn, 'encoder_5')
            layers.append(x)
            ch = ch // 2
            x = ops.resblock_up(x, ch, self.use_bias, self.training, self.sn, "decoder_5")
            layers.append(x)
            x = tf.concat([x, skips.pop(-1)], 3)
            # x = x + skips.pop(-1)
            x = ops.resblock_up(x, ch, self.use_bias, self.training, self.sn, "decoder_4")
            layers.append(x)
            ch = ch // 2
            x = tf.concat([x, skips.pop(-1)], 3)
            # x = x + skips.pop(-1)
            x = ops.resblock_up(x, ch, self.use_bias, self.training, self.sn, "decoder_3")
            layers.append(x)
            ch = ch // 2
            x = tf.concat([x, skips.pop(-1)], 3)
            # x = x + skips.pop(-1)
            x = ops.resblock_up(x, ch, self.use_bias, self.training, self.sn, "decoder_2")
            layers.append(x)
            # x = ops.self_attention_2(x, ch, sn=self.sn, scope='decoder_6')
            ch = ch // 2
            x = tf.concat([x, skips.pop(-1)], 3)
            # x = x + skips.pop(-1)
            x = ops.resblock_up(x, self.out_channels, self.use_bias, self.training, self.sn, "decoder_1")
            # layers.append(x)
            # x = ops.instance_norm(x, "instance_norm")
            # x = tf.nn.relu(x)
            # x = ops.conv(x, self.out_channels, kernel=3, stride=1, pad=1, use_bias=True, sn=self.sn,
            #              scope='output')
            # x = tf.nn.tanh(x)
        return x, layers

    def get_variables(self):
        variables = list()
        global_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        for i, var in enumerate(global_variables):
            var_names = var.name.split("/")
            if self.scope in var_names:
                variables.append(var)
        return variables


class Discriminator2:
    def __init__(self, ndf, n_layers=3, use_bias=True, training=True, scope=None):
        self.ndf = ndf
        self.n_layers = n_layers
        self.use_bias = use_bias
        self.training = training
        self.scope = scope

    def __call__(self, inputs):
        padding = [[0, 0], [1, 1], [1, 1], [0, 0]]
        layers = []
        # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]  padding="VALID",
        with tf.variable_scope(self.scope):
            with tf.variable_scope("layer_1"):
                padded = tf.pad(inputs, padding, mode="CONSTANT")
                convolved = ops.conv(padded, self.ndf, kernel=4, stride=2, pad=0, use_bias=self.use_bias,
                                     regularize=False)
                rectified = tf.nn.leaky_relu(convolved, 0.2)
                layers.append(convolved)
            # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
            # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
            # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
            for i in range(self.n_layers):
                with tf.variable_scope("layer_%d" % (i + 2)):
                    out_channels = self.ndf * min(2 ** (i + 1), 8)
                    stride = 1 if i == self.n_layers - 1 else 2  # last layer here has stride 1
                    padded = tf.pad(rectified, padding, mode="CONSTANT")
                    convolved = ops.conv(padded, out_channels, kernel=4, stride=stride, pad=0, use_bias=self.use_bias,
                                         regularize=False)
                    normalized = ops.batch_norm(convolved, 0.1, self.training)
                    rectified = tf.nn.leaky_relu(normalized, 0.2)
                    layers.append(convolved)
            # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
            with tf.variable_scope("layer_%d" % (self.n_layers + 2)):
                padded = tf.pad(rectified, padding, mode="CONSTANT")
                convolved = ops.conv(padded, 1, kernel=4, stride=1, pad=0, use_bias=self.use_bias, regularize=False)
                output = convolved
                # output = tf.nn.sigmoid(convolved)

        return output

    def get_variables(self):
        variables = list()
        global_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        for i, var in enumerate(global_variables):
            var_names = var.name.split("/")
            if self.scope in var_names:
                variables.append(var)
        return variables


class Discriminator:
    def __init__(self, ch, use_bias=False, sn=False, scope=None, training=True):
        self.ch = ch
        self.use_bias = use_bias
        self.scope = scope
        self.sn = sn
        self.training = training

    def __call__(self, x, return_feat=True):
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            ch = 2 * self.ch
            x = ops.resblock_down(x, ch, self.use_bias, self.training, self.sn, 'resblock_down_1')
            ch = 2 * ch
            x = ops.resblock_down(x, ch, self.use_bias, self.training, self.sn, 'resblock_down_2')
            # Non-Local Block
            x = ops.self_attention_2(x, ch, self.sn, 'self_attention')
            ch = 2 * ch
            x = ops.resblock_down(x, ch, self.use_bias, self.training, self.sn, 'resblock_down_4')
            x = ops.resblock_down(x, ch, self.use_bias, self.training, self.sn, 'resblock_down_8_0')
            ch = 2 * ch
            x = ops.resblock_down(x, ch, self.use_bias, self.training, self.sn, 'resblock_down_8_1')
            x = ops.resblock_down(x, ch, self.use_bias, self.training, self.sn, 'resblock_down_16')
            feat = x
            x = ops.resblock(x, ch, self.use_bias, self.training, self.sn, 'resblock')

            x = ops.relu(x)
            x = ops.global_sum_pooling(x)
            x = ops.fully_connected(x, units=1, sn=self.sn, scope='D_logit')

            if return_feat:
                return x, feat
            return x

    def get_variables(self):
        variables = list()
        global_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        for i, var in enumerate(global_variables):
            var_names = var.name.split("/")
            if self.scope in var_names or "shared" in var_names:
                variables.append(var)
        return variables