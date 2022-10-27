import numpy as np
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from scipy.linalg import sqrtm
from skimage.transform import resize
import os
from model.mycGAN import cGAN as fss_model
from util import utils1 as utils
import tensorflow as tf
from PIL import Image
import option
import matlab.engine
import FID

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_type', 'cufs', 'The type of dataset.')
tf.app.flags.DEFINE_string('sub_node', 'mycGAN_cufs_lmser', 'Directory to store model.')
tf.app.flags.DEFINE_string('gpu', '2', 'gpu device No.')

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

# scale an array of images to a new size
def scale_images(images, new_shape):
    images_list = list()
    for image in images:
        # resize with nearest neighbor interpolation
        new_image = resize(image, new_shape, 0)
        # store
        images_list.append(new_image)
    return asarray(images_list)


def load_txt2list(path):
    results = []
    f = open(path, mode='r')
    ls = f.readlines()
    for l in ls:
        l = l.strip("\n")
        if l.find(",") > 0:
            l = l.split(",")
        results.append(l)

    return results


def load_dataset(model_params):
    sketch_paths = load_txt2list("./data/{}_test_sketch_paths.txt".format(FLAGS.data_type.upper()))
    photo_paths = load_txt2list("./data/{}_test_photo_paths.txt".format(FLAGS.data_type.upper()))

    test_set = utils.DataLoader(
        sketch_paths,
        photo_paths,
        (400, 500),
        (572, 572),
        (512, 512),
        model_params.batch_size,
        model_params.is_training
    )

    return test_set


def load_checkpoint(sess, checkpoint_path):
    variables = tf.global_variables()
    variables_to_restore = [var for var in variables if var.name.find("vgg") < 0]
    saver = tf.train.Saver(variables_to_restore)
    ckpt = tf.train.get_checkpoint_state(checkpoint_path)
    print('Loading model %s' % ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)


def load_checkpoint2(sess, model_path):
    variables = tf.global_variables()
    saver = tf.train.Saver(var_list=variables)
    print('Loading model %s' % model_path)
    saver.restore(sess, model_path)


def load_images(photo_paths, mode):
    imgs = []
    for img_idx in range(len(photo_paths)):
        image_path = photo_paths[img_idx]
        image = Image.open(image_path).convert(mode=mode)
        imgs.append(image)

    return imgs


def evaluate_model(sess, model, test_set):
    length = test_set.sample_len
    for i in range(length):
        x1, x1_paths, x2, x2_paths, y, y_paths = test_set.get_batch(i)
        x1 = np.expand_dims(x1, 3)
        feed = {
            model.input_photo: y,
            model.input_sketch: x1
        }
        ### test sketch
        # """
        fake_y = sess.run(model.fake_y, feed)
        fake_y = np.clip(np.squeeze(fake_y), -1.0, 1.0)
        # fake_y = fake_y[6:506, 28:228]
        fake_y = fake_y[6:506, 56:456]
        fake_y = (fake_y + 1) * 127.5
        fake_y = np.rint(fake_y)

        path_split = y_paths[0].split("/")
        im_name = path_split[-3] + path_split[-1].replace("png", "jpg")
        im = Image.fromarray(np.uint8(fake_y))
        os.makedirs("/data/shengqingjie/outputs/Fss/Results/{}/Tmp/Sketch/".format(FLAGS.data_type.upper()),
                    exist_ok=True)
        im.save("/data/shengqingjie/outputs/Fss/Results/{}/Tmp/Sketch/{}".format(FLAGS.data_type.upper(), im_name))
        #     im = Image.fromarray(np.uint8(np.squeeze(x1)[3:253, 28:228]))
        #     im.save("/data/shengqingjie/outputs/Fss/Results/{}/GR/Sketch/{}".format(FLAGS.data_type.upper(), str(i+1)+".jpg"))


test_model_params = option.init()
test_model_params.drop_keep_prob = 1.0
test_model_params.is_training = 0
test_model_params.batch_size = 1
test_model = fss_model.Model(test_model_params)
tfconfig = tf.ConfigProto()
tfconfig.gpu_options.allow_growth = True
tfconfig.allow_soft_placement = True
sess = tf.InteractiveSession(config=tfconfig)
sess.run(tf.global_variables_initializer())
load_checkpoint2(sess, "/data/shengqingjie/outputs/Fss/snapshot/{}/best_model/fss_model".format(FLAGS.sub_node))
# load_checkpoint2(sess, "/data/shengqingjie/outputs/Fss/snapshot/{}/cufsf-77750".format(FLAGS.sub_node))
test_set = load_dataset(test_model_params)
evaluate_model(sess, test_model, test_set)

