from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from model.mycGAN.net import Generator, Discriminator2 as Discriminator
from model.mycGAN.vgg import Vgg16
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
        self.Vgg = Vgg16(scope="Vgg16", trainable=False)
        self.selected_layer = [0, 2, 10]  # relu1_1, relu3_1, relu5_1
        if self.hps.is_training:
            self.start_decay_steps = self.hps.start_decay_epochs * self.hps.steps_per_epoch
            self.decay_steps = self.hps.decay_epochs * self.hps.steps_per_epoch
            self.Dy = Discriminator(ndf=64, n_layers=3, use_bias=False, training=True, scope="Dy")
            self.Fake_Y_Pool = ImagePool(self.hps.pool_size)

    def build_model(self, name):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            self.build_model_basic()
        if self.hps.is_training:
            with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.build_model_extra(name)
            with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
                self.build_losses()
                self.optimize_model()

    def build_model_basic(self):
        self.fake_y, _ = self.G(self.input_x)

    def build_model_extra(self, name):
        self.perceptual_layers = list()
        self.real_layers = list()
        fake_y = tf.tile(self.fake_y, [1, 1, 1, 3])
        input_y = tf.tile(self.input_y, [1, 1, 1, 3])
        # It was 93.5940, 104.7624, 129.1863 before dividing by 255
        MEAN_RGB = np.array(self.hps.batch_size * [self.hps.out_size[0] * [self.hps.out_size[1] * [[0.367035294117647, 0.41083294117647057, 0.5066129411764705]]]])
        STD_RGB = np.array(self.hps.batch_size * [self.hps.out_size[0] * [self.hps.out_size[1] * [[1, 1, 1]]]])
        fake_y = ((fake_y + 1) / 2 - MEAN_RGB) / STD_RGB
        input_y = ((input_y + 1) / 2 - MEAN_RGB) / STD_RGB
        out1 = self.Vgg(fake_y)
        out2 = self.Vgg(input_y)
        for i in self.selected_layer:
            self.perceptual_layers.append(out1[i])
            self.real_layers.append(out2[i])

        self.real_yx = tf.concat([self.input_y, self.input_x], 3)
        self.real_xy = tf.concat([self.input_x, self.input_y], 3)
        self.fake_xy = tf.concat([self.input_x, self.fake_y], 3)
        self.pool_xy = self.Fake_Y_Pool.query(self.fake_xy)
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            self.pred_real_xy = self.Dy(self.real_xy)
            self.pred_pool_xy = self.Dy(self.pool_xy)
            self.pred_fake_xy = self.Dy(self.fake_xy)

    def build_losses(self):
        self.loss_perceptual = 0
        for i in range(len(self.perceptual_layers)):
            self.loss_perceptual += tf.reduce_mean(tf.square(self.perceptual_layers[i] - self.real_layers[i]))
        self.loss_perceptual = self.loss_perceptual / len(self.perceptual_layers)

        self.loss_Dy = ops.discriminator_loss(self.hps.gan_type, self.pred_real_xy, self.pred_pool_xy)
        self.loss_G = self.hps.l1_lambda * tf.reduce_mean(tf.abs(self.fake_y - self.input_y)) + \
                      ops.generator_loss(self.hps.gan_type, self.pred_fake_xy) + 5 * self.loss_perceptual
                      # 1 * (self.loss_x_cyclic + self.loss_y_content + self.loss_y_style) #

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

        self.g_lr = 0.0001
        self.d_lr = 0.0002

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.Dy_optimizer = make_optimizer(self.loss_Dy, self.d_lr,
                                              self.Dy.get_variables(),
                                              optimizer="Adam", name='Adam_Dy')
            self.G_optimizer = make_optimizer(self.loss_G, self.g_lr,
                                              self.G.get_variables(),
                                              optimizer="Adam", name='Adam_G')
            self.step_op = tf.assign(self.global_step, self.global_step + 1)
