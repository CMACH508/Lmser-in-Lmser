from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from model.mycGAN.net import Generator, Discriminator2 as Discriminator
from util.image_pool import ImagePool
import model.mycGAN.ops as ops


class Model(object):

    def __init__(self, hps):
        self.hps = hps
        self.config_model()
        self.build_model("fss")

    def config_model(self):
        self.input_photo = tf.placeholder(
            dtype=tf.float32,
            shape=[self.hps.batch_size, self.hps.out_size[0], self.hps.out_size[1], 3])
        self.input_sketch = tf.placeholder(
            dtype=tf.float32,
            shape=[self.hps.batch_size, self.hps.out_size[0], self.hps.out_size[1], 1])

        # Normalizing image
        # [N, H, W, C], [-1, 1]
        self.input_x = self.input_photo / 127.5 - 1
        self.input_y = self.input_sketch / 127.5 - 1

        self.G = Generator(out_channels=1, ch=64, use_bias=False, scope="G", training=self.hps.is_training)
        self.F = Generator(out_channels=3, ch=64, use_bias=False, scope="F", training=self.hps.is_training)
        if self.hps.is_training:
            self.start_decay_steps = self.hps.start_decay_epochs * self.hps.steps_per_epoch
            self.decay_steps = self.hps.decay_epochs * self.hps.steps_per_epoch
            self.Dy = Discriminator(ndf=64, n_layers=3, use_bias=False, training=True, scope="Dy")
            self.Dx = Discriminator(ndf=64, n_layers=3, use_bias=False, training=True, scope="Dx")
            self.Fake_X_Pool = ImagePool(self.hps.pool_size)
            self.Fake_Y_Pool = ImagePool(self.hps.pool_size)

    def build_model(self, name):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            self.build_model_basic()
            if self.hps.is_training:
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.build_model_extra()
                self.build_losses()
                self.optimize_model()

    def build_model_basic(self):
        self.fake_y, self.g_real_layers = self.G(self.input_x)
        self.fake_x, self.f_real_layers = self.F(self.input_y)

    def build_model_extra(self):
        self.cyclic_x, self.f_fake_layers = self.F(self.fake_y)
        # self.cyclic_y, self.g_fake_layers = self.G(self.fake_x)

        self.real_yx = tf.concat([self.input_y, self.input_x], 3)
        self.fake_yx = tf.concat([self.input_y, self.fake_x], 3)
        self.pool_yx = self.Fake_X_Pool.query(self.fake_yx)
        self.pred_real_yx = self.Dx(self.real_yx)
        self.pred_pool_yx = self.Dx(self.pool_yx)
        self.pred_fake_yx = self.Dx(self.fake_yx)
        self.real_xy = tf.concat([self.input_x, self.input_y], 3)
        self.fake_xy = tf.concat([self.input_x, self.fake_y], 3)
        self.pool_xy = self.Fake_Y_Pool.query(self.fake_xy)
        self.pred_real_xy = self.Dy(self.real_xy)
        self.pred_pool_xy = self.Dy(self.pool_xy)
        self.pred_fake_xy = self.Dy(self.fake_xy)

    def build_losses(self):
        # length = len(self.f_real_layers)
        # ds1 = None
        # for i in range(length):
        # for i in [4]:
        #     d = tf.reduce_mean(tf.abs(self.f_real_layers[i] - self.f_fake_layers[i]))
        #     d = tf.expand_dims(d, axis=0)
        #     if ds1 is None:
        #         ds1 = d
        #     else:
        #         ds1 = tf.concat([ds1, d], axis=0)
        self.loss_f_layers = tf.reduce_mean(tf.abs(self.f_real_layers[4] - self.f_fake_layers[4]))
        # self.loss_g_layers = tf.reduce_mean(tf.abs(self.g_real_layers[4] - self.g_fake_layers[4]))

        # ds2 = None
        # self.loss_lmser = 0
        # for i in range(length):
        # for i in [2, 4, 6]:
        #     d = tf.reduce_mean(tf.abs(self.g_real_layers[i] - self.f_fake_layers[-i - 1]))
        #     self.loss_lmser += d
            # d = tf.expand_dims(d, axis=0)
            # if ds2 is None:
            #     ds2 = d
            # else:
            #     ds2 = tf.concat([ds2, d], axis=0)
        # self.loss_lmser = tf.reduce_sum(ds2)

        self.loss_x_cyclic = tf.reduce_mean(tf.abs(self.cyclic_x - self.input_x))
        # self.loss_y_cyclic = tf.reduce_mean(tf.abs(self.cyclic_y - self.input_y))

        self.loss_Dx = ops.discriminator_loss(self.hps.gan_type, self.pred_real_yx, self.pred_pool_yx)
        self.loss_F = self.hps.l1_lambda * tf.reduce_mean(tf.abs(self.fake_x - self.input_x)) + \
                      ops.generator_loss(self.hps.gan_type, self.pred_fake_yx) #+ 10 * self.loss_lmser +\
                      # 10 * self.loss_y_cyclic + 10 * self.loss_g_layers

        self.loss_Dy = ops.discriminator_loss(self.hps.gan_type, self.pred_real_xy, self.pred_pool_xy)
        self.loss_G = self.hps.l1_lambda * tf.reduce_mean(tf.abs(self.fake_y - self.input_y)) + \
                      ops.generator_loss(self.hps.gan_type, self.pred_fake_xy) + \
                      5 * (self.loss_x_cyclic + self.loss_f_layers)

    def optimize_model(self):
        def make_optimizer(loss, lr, variables, optimizer="Adam", name='Adam'):
            # tf.summary.scalar('learning_rate/{}'.format(name), learning_rate)
            step = tf.Variable(0, name=name + "_step", trainable=False)
            if optimizer == "Adam":
                learning_step = (
                    tf.train.AdamOptimizer(lr, beta1=self.hps.beta1, name=name)
                        .minimize(loss, global_step=step, var_list=variables)
                )
            elif optimizer == "RMSProp":
                learning_step = (
                    tf.train.RMSPropOptimizer(lr, name=name)
                        .minimize(loss, global_step=step, var_list=variables)
                )
            elif optimizer == "GD" or optimizer == "SGD":
                learning_step = (
                    tf.train.GradientDescentOptimizer(lr, name=name)
                        .minimize(loss, global_step=step, var_list=variables)
                )
            else:
                raise Exception("Unexpected optimizer: {}".format(name))

            return learning_step

        # self.g_lr = (
        #     tf.where(
        #         tf.greater_equal(self.global_step, self.start_decay_steps),
        #         tf.train.polynomial_decay(self.hps.lr, self.global_step - self.start_decay_steps,
        #                                   self.decay_steps, self.hps.min_learning_rate,
        #                                   power=1.0),
        #         self.hps.lr
        #     )
        # )

        # self.g_lr = self.hps.g_lr
        # self.d_lr = self.hps.d_lr
        self.g_lr = 0.0001
        self.d_lr = 0.0002

        g_vars = self.G.get_variables()
        f_vars = self.F.get_variables()
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.Dx_optimizer = make_optimizer(self.loss_Dx, self.d_lr,
                                               self.Dx.get_variables(),
                                               optimizer="Adam", name='Adam_Dx')
            self.Dy_optimizer = make_optimizer(self.loss_Dy, self.d_lr,
                                               self.Dy.get_variables(),
                                               optimizer="Adam", name='Adam_Dy')
            self.G_optimizer = make_optimizer(self.loss_G, self.g_lr,
                                              g_vars,
                                              optimizer="Adam", name='Adam_G')
            self.F_optimizer = make_optimizer(self.loss_F, self.g_lr,
                                              f_vars,
                                              optimizer="Adam", name='Adam_F')
            self.step_op = tf.assign(self.global_step, self.global_step + 1)
