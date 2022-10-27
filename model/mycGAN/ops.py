import tensorflow as tf
import tensorflow.compat.v2 as tf_v2


##################################################################################
# Regularization
##################################################################################
def orthogonal_regularizer(scale):
    """ Defining the Orthogonal regularizer and return the function at last to be used in Conv layer as kernel regularizer"""

    def ortho_reg(w):
        """ Reshaping the matrxi in to 2D tensor for enforcing orthogonality"""
        _, _, _, c = w.get_shape().as_list()

        w = tf.reshape(w, [-1, c])

        """ Declaring a Identity Tensor of appropriate size"""
        identity = tf.eye(c)

        """ Regularizer Wt*W - I """
        w_transpose = tf.transpose(w)
        w_mul = tf.matmul(w_transpose, w)
        reg = tf.subtract(w_mul, identity)

        """Calculating the Loss Obtained"""
        ortho_loss = tf.nn.l2_loss(reg)

        return scale * ortho_loss

    return ortho_reg


def orthogonal_regularizer_fully(scale):
    """ Defining the Orthogonal regularizer and return the function at last to be used in Fully Connected Layer """

    def ortho_reg_fully(w):
        """ Reshaping the matrix in to 2D tensor for enforcing orthogonality"""
        _, c = w.get_shape().as_list()

        """Declaring a Identity Tensor of appropriate size"""
        identity = tf.eye(c)
        w_transpose = tf.transpose(w)
        w_mul = tf.matmul(w_transpose, w)
        reg = tf.subtract(w_mul, identity)

        """ Calculating the Loss """
        ortho_loss = tf.nn.l2_loss(reg)

        return scale * ortho_loss

    return ortho_reg_fully


##################################################################################
# Initialization
##################################################################################

# Xavier : tf_contrib.layers.xavier_initializer()
# He : tf_contrib.layers.variance_scaling_initializer()
# Normal : tf.random_normal_initializer(mean=0.0, stddev=0.02)
# Truncated_normal : tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
# Orthogonal : tf.orthogonal_initializer(1.0) / relu = sqrt(2), the others = 1.0
# l2_decay : tf_contrib.layers.l2_regularizer(0.0001)
# orthogonal_regularizer : orthogonal_regularizer(0.0001) / orthogonal_regularizer_fully(0.0001)

weight_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
weight_regularizer = orthogonal_regularizer(0.0001)
weight_regularizer_fully = orthogonal_regularizer_fully(0.0001)


# Regularization only G in BigGAN

##################################################################################
# Layer
##################################################################################

# pad = ceil[ (kernel - stride) / 2 ]

def conv(x, channels, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True, sn=False, regularize=False,
         scope='conv_0', direction=True):
    with tf.variable_scope(scope):
        if pad > 0:
            h = x.get_shape().as_list()[1]
            if h % stride == 0:
                pad = pad * 2
            else:
                pad = max(kernel - (h % stride), 0)

            pad_top = pad // 2
            pad_bottom = pad - pad_top
            pad_left = pad // 2
            pad_right = pad - pad_left

            if pad_type == 'zero':
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
            if pad_type == 'reflect':
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode='REFLECT')

        if sn:
            if regularize:
                w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels],
                                    initializer=weight_init,
                                    regularizer=weight_regularizer)
            else:
                w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels],
                                    initializer=weight_init,
                                    regularizer=None)

            x = tf.nn.conv2d(input=x, filter=spectral_norm(w),
                             strides=[1, stride, stride, 1], padding='VALID')
            if use_bias:
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, bias)

        else:
            if regularize:
                x = tf.layers.conv2d(inputs=x, filters=channels,
                                     kernel_size=kernel, kernel_initializer=weight_init,
                                     kernel_regularizer=weight_regularizer,
                                     strides=stride, use_bias=use_bias)
            else:
                x = tf.layers.conv2d(inputs=x, filters=channels,
                                     kernel_size=kernel, kernel_initializer=weight_init,
                                     kernel_regularizer=None,
                                     strides=stride, use_bias=use_bias)

        return x


def deconv(x, channels, kernel=4, stride=2, padding='SAME', use_bias=True, sn=False, scope='deconv_0'):
    with tf.variable_scope(scope):
        x_shape = x.get_shape().as_list()

        if padding == 'SAME':
            output_shape = [x_shape[0], x_shape[1] * stride, x_shape[2] * stride, channels]

        else:
            output_shape = [x_shape[0], x_shape[1] * stride + max(kernel - stride, 0),
                            x_shape[2] * stride + max(kernel - stride, 0), channels]

        if sn:
            w = tf.get_variable("kernel", shape=[kernel, kernel, channels, x.get_shape()[-1]], initializer=weight_init,
                                regularizer=weight_regularizer)
            x = tf.nn.conv2d_transpose(x, filter=spectral_norm(w), output_shape=output_shape,
                                       strides=[1, stride, stride, 1], padding=padding)

            if use_bias:
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, bias)

        else:
            x = tf.layers.conv2d_transpose(inputs=x, filters=channels,
                                           kernel_size=kernel, kernel_initializer=weight_init,
                                           kernel_regularizer=weight_regularizer,
                                           strides=stride, padding=padding, use_bias=use_bias)

        return x


def fully_connected(x, units, use_bias=True, sn=False, regularize=True, scope='fully_0'):
    with tf.variable_scope(scope):
        x = flatten(x)
        shape = x.get_shape().as_list()
        channels = shape[-1]

        if sn:
            if regularize:
                w = tf.get_variable("kernel", [channels, units], tf.float32, initializer=weight_init,
                                    regularizer=weight_regularizer_fully)
            else:
                w = tf.get_variable("kernel", [channels, units], tf.float32, initializer=weight_init, regularizer=None)

            if use_bias:
                bias = tf.get_variable("bias", [units], initializer=tf.constant_initializer(0.0))

                x = tf.matmul(x, spectral_norm(w)) + bias
            else:
                x = tf.matmul(x, spectral_norm(w))

        else:
            if regularize:
                x = tf.layers.dense(x, units=units, kernel_initializer=weight_init,
                                    kernel_regularizer=weight_regularizer_fully, use_bias=use_bias)
            else:
                x = tf.layers.dense(x, units=units, kernel_initializer=weight_init,
                                    kernel_regularizer=None, use_bias=use_bias)

        return x


def flatten(x):
    return tf.layers.flatten(x)


def hw_flatten(x):
    return tf.reshape(x, shape=[x.shape[0], -1, x.shape[-1]])


##################################################################################
# Residual-block, Self-Attention-block
##################################################################################

def resblock(x_init, channels, use_bias=True, training=True, sn=False, scope='resblock'):
    with tf.variable_scope(scope):
        with tf.variable_scope('res1'):
            x = conv(x_init, channels, kernel=3, stride=1, pad=1, use_bias=use_bias, sn=sn)
            x = batch_norm(x, training=training)
            x = lrelu(x)

        with tf.variable_scope('res2'):
            x = conv(x, channels, kernel=3, stride=1, pad=1, use_bias=use_bias, sn=sn)
            x = batch_norm(x, training=training)

        return x + x_init


def resblock_up(x_init, channels, use_bias=True, training=True, sn=False, scope='resblock_up'):
    with tf.variable_scope(scope):
        with tf.variable_scope('res1'):
            # x = batch_norm(x_init, training=training)
            x = instance_norm(x_init)
            x = relu(x)
            # x = deconv(x, channels, kernel=3, stride=2, use_bias=use_bias, sn=sn)
            x = up_sample(x, 2)
            x = conv(x, channels, kernel=3, stride=1, pad=1, use_bias=use_bias, sn=sn)

        with tf.variable_scope('res2'):
            # x = batch_norm(x, training=training)
            x = instance_norm(x)
            x = relu(x)
            x = conv(x, channels, kernel=3, stride=1, pad=1, use_bias=use_bias, sn=sn)

        with tf.variable_scope('skip'):
            x_init = up_sample(x_init, 2)
            x_init = conv(x_init, channels, kernel=1, stride=1, pad=0, use_bias=use_bias, sn=sn)

    return x + x_init


def resblock_up_condition(x_init, z, channels, use_bias=True, is_training=True, sn=False, scope='resblock_up'):
    with tf.variable_scope(scope):
        with tf.variable_scope('res1'):
            x = condition_batch_norm(x_init, z, is_training)
            x = relu(x)
            x = deconv(x, channels, kernel=3, stride=2, use_bias=use_bias, sn=sn)

        with tf.variable_scope('res2'):
            x = condition_batch_norm(x, z, is_training)
            x = relu(x)
            x = deconv(x, channels, kernel=3, stride=1, use_bias=use_bias, sn=sn)

        with tf.variable_scope('skip'):
            x_init = deconv(x_init, channels, kernel=3, stride=2, use_bias=use_bias, sn=sn)

    return x + x_init


def resblock_down(x_init, channels, use_bias=True, training=True, sn=False, scope='resblock_down'):
    with tf.variable_scope(scope):
        with tf.variable_scope('res1'):
            # x = batch_norm(x_init, training=training)
            x = instance_norm(x_init)
            x = lrelu(x)
            x = conv(x, channels, kernel=3, stride=1, pad=1, use_bias=use_bias, sn=sn)

        with tf.variable_scope('res2'):
            # x = batch_norm(x, training=training)
            x = instance_norm(x)
            x = lrelu(x)
            x = conv(x, channels, kernel=3, stride=1, pad=1, use_bias=use_bias, sn=sn)
            x = down_sample(x)

        with tf.variable_scope('skip'):
            x_init = conv(x_init, channels, kernel=1, stride=1, pad=0, use_bias=use_bias, sn=sn)
            x_init = down_sample(x_init)

    return x + x_init


def self_attention(x, channels, sn=False, scope='self_attention'):
    with tf.variable_scope(scope):
        f = conv(x, channels // 8, kernel=1, stride=1, sn=sn, scope='f_conv')  # [bs, h, w, c']
        g = conv(x, channels // 8, kernel=1, stride=1, sn=sn, scope='g_conv')  # [bs, h, w, c']
        h = conv(x, channels, kernel=1, stride=1, sn=sn, scope='h_conv')  # [bs, h, w, c]

        # N = h * w
        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # # [bs, N, N]

        beta = tf.nn.softmax(s)  # attention map

        o = tf.matmul(beta, hw_flatten(h))  # [bs, N, C]
        gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

        o = tf.reshape(o, shape=x.shape)  # [bs, h, w, C]
        x = gamma * o + x

    return x


def self_attention_2(x, channels, sn=False, scope='self_attention'):
    with tf.variable_scope(scope):
        f = conv(x, channels // 8, kernel=1, stride=1, sn=sn, scope='f_conv')  # [bs, h, w, c']
        f = max_pooling(f)

        g = conv(x, channels // 8, kernel=1, stride=1, sn=sn, scope='g_conv')  # [bs, h, w, c']

        h = conv(x, channels // 2, kernel=1, stride=1, sn=sn, scope='h_conv')  # [bs, h, w, c]
        h = max_pooling(h)

        # N = h * w
        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # # [bs, N, N]

        beta = tf.nn.softmax(s)  # attention map

        # TODO: check that softmax along the last dimension was the correct one

        o = tf.matmul(beta, hw_flatten(h))  # [bs, N, C]
        gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

        o = tf.reshape(o, shape=[x.shape[0], x.shape[1], x.shape[2], channels // 2])  # [bs, h, w, C]
        o = conv(o, channels, kernel=1, stride=1, sn=sn, scope='attn_conv')
        x = gamma * o + x

    return x


##################################################################################
# Sampling
##################################################################################

def global_avg_pooling(x):
    gap = tf.reduce_mean(x, axis=[1, 2])

    return gap


def global_sum_pooling(x):
    gsp = tf.reduce_sum(x, axis=[1, 2])

    return gsp


def max_pooling(x):
    x = tf.layers.max_pooling2d(x, pool_size=2, strides=2, padding='SAME')
    return x


def down_sample(x, scale=2):
    x = tf.layers.average_pooling2d(x, pool_size=scale, strides=scale, padding="SAME")
    return x


def up_sample(x, scale_factor=2):
    _, h, w, _ = x.get_shape().as_list()
    new_size = [h * scale_factor, w * scale_factor]
    return tf.image.resize(x, size=new_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)


##################################################################################
# Activation function
##################################################################################

def lrelu(x, alpha=0.2):
    return tf.nn.leaky_relu(x, alpha)


def relu(x):
    return tf.nn.relu(x)


def tanh(x):
    return tf.tanh(x)


##################################################################################
# Normalization function
##################################################################################

def batch_norm(inputs, momentum=0.9, training=True, scope='batch_norm'):
    # with tf.variable_scope(scope):
    #     c = inputs.get_shape().as_list()[-1]
    #     scale = tf.get_variable('scale', [c], initializer=tf.constant_initializer(0.1))
    #     offset = tf.get_variable('offset', [c])
    #
    #     running_mean = tf.get_variable('running_mean', [c], initializer=tf.zeros_initializer(), trainable=False)
    #     running_var = tf.get_variable('running_var', [c], initializer=tf.ones_initializer(), trainable=False)
    #     batch_mean, batch_var = tf.nn.moments(inputs, axes=[0, 1, 2])
    #     train_mean_op = tf.assign(running_mean, running_mean * momentum + batch_mean * (1 - momentum))
    #     train_var_op = tf.assign(running_var, running_var * momentum + batch_var * (1 - momentum))
    #     epsilon = 1e-5
    #
    #     def batch_statistics():
    #         with tf.control_dependencies([train_mean_op, train_var_op]):
    #             return tf.nn.batch_normalization(inputs, batch_mean, batch_var, offset, scale, epsilon)
    #
    #     def population_statistics():
    #         return tf.nn.batch_normalization(inputs, running_mean, running_var, offset, scale, epsilon)
    #
    #     training = tf.cast(training, tf.bool)
    #     return tf.cond(training, batch_statistics, population_statistics)
    return tf.layers.batch_normalization(inputs, momentum=momentum, epsilon=1e-05, training=training, name=scope)


def condition_batch_norm(x, z, is_training=True, scope='batch_norm'):
    with tf.variable_scope(scope):
        _, _, _, c = x.get_shape().as_list()
        decay = 0.9
        epsilon = 1e-05

        test_mean = tf.get_variable("pop_mean", shape=[c], dtype=tf.float32, initializer=tf.constant_initializer(0.0),
                                    trainable=False)
        test_var = tf.get_variable("pop_var", shape=[c], dtype=tf.float32, initializer=tf.constant_initializer(1.0),
                                   trainable=False)

        beta = fully_connected(z, units=c, scope='beta')
        gamma = fully_connected(z, units=c, scope='gamma')

        beta = tf.reshape(beta, shape=[-1, 1, 1, c])
        gamma = tf.reshape(gamma, shape=[-1, 1, 1, c])

        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2])
        # Update exponential moving averages of the batch mean and var
        ema_mean = tf.assign(test_mean, test_mean * decay + batch_mean * (1 - decay))
        ema_var = tf.assign(test_var, test_var * decay + batch_var * (1 - decay))
        if is_training:
            with tf.control_dependencies([ema_mean, ema_var]):
                return tf.nn.batch_normalization(x, batch_mean, batch_var, beta, gamma, epsilon)
        else:
            return tf.nn.batch_normalization(x, test_mean, test_var, beta, gamma, epsilon)


def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """

        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm


def instance_norm(inputs, scope="instance_norm"):
    with tf.variable_scope(scope):
        c = inputs.get_shape()[-1]
        scale = tf.get_variable("scale", [c], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32),
                                trainable=True)
        offset = tf.get_variable("offset", [c], initializer=tf.constant_initializer(0.0), trainable=True)
        mean, variance = tf.nn.moments(inputs, axes=[1, 2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.math.rsqrt(variance + epsilon)
        normalized = (inputs - mean) * inv

        return scale * normalized + offset


def norm(inputs, scope="norm", norm_type="instance_norm", training=True):
    assert norm_type in ["instance_norm", "batch_norm"]
    if norm_type == "instance_norm":
        return instance_norm(inputs, scope)
    elif norm_type == "batch_norm":
        return batch_norm(inputs, 0.9, training, scope)


##################################################################################
# Loss function
##################################################################################

def discriminator_loss(loss_func, real, fake):
    real_loss = 0
    fake_loss = 0

    if loss_func.__contains__('wgan'):
        real_loss = -tf.reduce_mean(real)
        fake_loss = tf.reduce_mean(fake)

    if loss_func == 'lsgan':
        real_loss = tf.reduce_mean(tf.squared_difference(real, 1.0))
        fake_loss = tf.reduce_mean(tf.square(fake))

    if loss_func == 'gan' or loss_func == 'dragan':
        real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real), logits=real))
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake), logits=fake))

    if loss_func == 'hinge':
        real_loss = tf.reduce_mean(relu(1.0 - real))
        fake_loss = tf.reduce_mean(relu(1.0 + fake))

    loss = real_loss + fake_loss

    return loss


def generator_loss(loss_func, fake):
    fake_loss = 0

    if loss_func.__contains__('wgan'):
        fake_loss = -tf.reduce_mean(fake)

    if loss_func == 'lsgan':
        fake_loss = tf.reduce_mean(tf.squared_difference(fake, 1.0))

    if loss_func == 'gan' or loss_func == 'dragan':
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake), logits=fake))

    if loss_func == 'hinge':
        fake_loss = -tf.reduce_mean(fake)

    loss = fake_loss

    return loss

def total_variation(x):
    """
    Total Variation Loss.
    """
    return tf.reduce_sum(tf.square(x[:, :-1, :, :] - x[:, 1:, :, :])
            ) + tf.reduce_sum(tf.square(x[:, :, :-1, :] - x[:, :, 1:, :]))


def style_loss(source, target, kernel=5, stride=4, layer="layer"):
    ksizes = [1, kernel, kernel, 1]
    strides = [1, stride, stride, 1]
    rates = [1, 1, 1, 1]

    x_patches = tf.extract_image_patches(source, ksizes=ksizes, strides=strides, rates=rates, padding="SAME")
    y_patches = tf.extract_image_patches(target, ksizes=ksizes, strides=strides, rates=rates, padding="SAME")

    # with tf.variable_scope(layer, reuse=tf.AUTO_REUSE):
    #     w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels],
    #                         initializer=)
    #
    # x = tf.nn.conv2d(input=x, filter=w, strides=[1, 1, 1, 1], padding='VALID')
    # # distance = tf.losses.cosine_distance(x_patches, y_patches, axis=3)
    #
    # distance =

    # shape = x_patches.get_shape()
    # x_patches = tf.reshape(x_patches, [shape[0], shape[1] * shape[2], shape[3]])
    # y_patches = tf.reshape(y_patches, [shape[0], shape[1] * shape[2], shape[3]])
    #
    # for i in range(shape[1] * shape[2]):
    #     x_patch = x_patches[:, i, :]
    #     x_patch_tiled = tf.tile(tf.expand_dims(x_patch, axis=1), [1, shape[1] * shape[2], 1])
    #     distances = tf.losses.cosine_distance(x_patch_tiled, y_patches, axis=2, reduction="none")
    #     distances = tf.squeeze(distances, axis=2)
    #     indice = tf.argmin(distances, axis=1)
    #     loss = tf.reduce_mean(tf.abs(x_patch - y_patches[:, indice, :]))

    # return distance
