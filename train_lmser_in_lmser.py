from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import time
import copy

import numpy as np
from PIL import Image
import tensorflow as tf
from pytorch_fid import fid_score as FID

from model.mycGAN import cGAN as fss_model
from util import utils as utils
import option

tf.logging.set_verbosity(tf.logging.INFO)

###重置tensorflow张量图###
def reset_graph():
    """Closes the current default session and resets the graph."""
    sess = tf.get_default_session()
    if sess:
        sess.close()
    tf.reset_default_graph()

###加载人脸照片数据###
def load_images(data_dir):
    file_paths = []
    categories = os.listdir(data_dir)
    for category in categories:
        category_d = os.path.join(data_dir, category)
        samples = os.listdir(category_d)
        for i in range(len(samples)):
            file_paths.append(os.path.join(category_d, samples[i]))

    return file_paths

###加载素描图片数据###
def load_sketches(sketch_dir):
    file_paths = []
    categories = os.listdir(sketch_dir)
    for category in categories:
        category_d = os.path.join(sketch_dir, category)
        samples = os.listdir(category_d)
        for i in range(len(samples)):
            file_paths.append(os.path.join(category_d, samples[i]))

    return file_paths

###将list保存为txt文件，option.py中的data_load_mode参数为create时调用，目的是将数据集划分情况持久化###
def save_list2txt(l, path):
    f = open(path, mode='w')
    for i in l:
        if isinstance(i, list):
            i = ','.join(i)
        f.write(i)
        f.write("\n")
    f.close()

###将txt加载成list，训练时的数据集路径列表，读取save_list2txt方法中持久化的数据集划分文件，用于训练或测试###
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

###写日志文件###
def write_log(log, path):
    f = open(path, mode='a')
    f.write(log)
    f.write("\n")
    f.close()

###获取目录的绝对路径###
def abs_listdir(dir):
    abs_paths = list()
    for path in os.listdir(dir):
        abs_paths.append(os.path.join(dir, path))

    abs_paths.sort()
    return abs_paths

###数据集加载###
def load_dataset(params):
    data_type = params.data_type.upper()
    data_dir = params.data_dir

    ###数据集划分模式为create，通常为第一次训练时尚未持久化数据集划分列表的情况###
    if params.data_load_mode == "create":
        if data_type == 'CUFSF':
            train_sketch_dir = os.path.join(data_dir, "CUFSF/Training Sketches/")
            train_photo_dir = os.path.join(data_dir, "CUFSF/Training Photos/")
            test_sketch_dir = os.path.join(data_dir, "CUFSF/Testing Sketches/")
            test_photo_dir = os.path.join(data_dir, "CUFSF/Testing Photos/")

            train_sketch_paths = abs_listdir(train_sketch_dir)
            train_photo_paths = abs_listdir(train_photo_dir)
            test_sketch_paths = abs_listdir(test_sketch_dir)
            test_photo_paths = abs_listdir(test_photo_dir)
        elif data_type == 'CUFS':
            train_sketch_paths = list()
            train_photo_paths = list()
            test_sketch_paths = list()
            test_photo_paths = list()
            sub_data_list = ["CUHK", "AR", "XM2VTS"]
            for i, sub_data in enumerate(sub_data_list):
                train_sketch_dir = os.path.join(data_dir, "{}/Training Sketches/".format(sub_data))
                train_photo_dir = os.path.join(data_dir, "{}/Training Photos/".format(sub_data))
                test_sketch_dir = os.path.join(data_dir, "{}/Testing Sketches/".format(sub_data))
                test_photo_dir = os.path.join(data_dir, "{}/Testing Photos/".format(sub_data))

                train_sketch_paths += abs_listdir(train_sketch_dir)
                train_photo_paths += abs_listdir(train_photo_dir)
                test_sketch_paths += abs_listdir(test_sketch_dir)
                test_photo_paths += abs_listdir(test_photo_dir)
        else:
            raise Exception('Unknown data type:', data_type)

        save_list2txt(train_sketch_paths, "./data/{}_train_sketch_paths.txt".format(data_type))
        save_list2txt(train_photo_paths, "./data/{}_train_photo_paths.txt".format(data_type))
        save_list2txt(test_sketch_paths, "./data/{}_test_sketch_paths.txt".format(data_type))
        save_list2txt(test_photo_paths, "./data/{}_test_photo_paths.txt".format(data_type))
    ###绝大多数情况下的选择###
    elif params.data_load_mode == "load":
        if data_type in ['CUFS', 'CUFSF']:
            train_sketch_paths = load_txt2list("./data/{}_train_sketch_paths.txt".format(data_type))
            train_photo_paths = load_txt2list("./data/{}_train_photo_paths.txt".format(data_type))
            test_sketch_paths = load_txt2list("./data/{}_test_sketch_paths.txt".format(data_type))
            test_photo_paths = load_txt2list("./data/{}_test_photo_paths.txt".format(data_type))
        elif data_type == 'BOTH':
            train_sketch_paths = []
            train_photo_paths = []
            test_sketch_paths = []
            test_photo_paths = []
            for i in ['CUFS', 'CUFSF']:
                train_sketch_paths += load_txt2list("./data/{}_train_sketch_paths.txt".format(i))
                train_photo_paths += load_txt2list("./data/{}_train_photo_paths.txt".format(i))
                test_sketch_paths += load_txt2list("./data/{}_test_sketch_paths.txt".format(i))
                test_photo_paths += load_txt2list("./data/{}_test_photo_paths.txt".format(i))
        else:
            raise Exception('Unexpected data type:', data_type)
    else:
        raise Exception('Unexpected data loading mode:', params.data_load_mode)

    len1 = len(train_sketch_paths)
    len2 = len(test_sketch_paths)
    len3 = len(train_photo_paths)
    len4 = len(test_photo_paths)
    print('Loaded datasets from {}. sketch: {}(train: {}, test: {}), photo: {}(train: {}, test: {})'
          .format(data_type, len1 + len2, len1, len2, len3 + len4, len3, len4))

    train_model_params = copy.deepcopy(params)
    train_model_params.steps_per_epoch = int(len1 / params.batch_size)
    eval_model_params = copy.deepcopy(train_model_params)
    eval_model_params.drop_keep_prob = 1.0
    eval_model_params.is_training = False
    eval_model_params.batch_size = 1

    ###训练集的数据加载器###
    train_set = utils.DataLoader(
        train_sketch_paths,
        train_photo_paths,
        train_model_params.in_size,
        train_model_params.pad_size,
        train_model_params.out_size,
        train_model_params.batch_size,
        train_model_params.is_training
    )
    ###测试集的数据加载器###
    test_set = utils.DataLoader(
        test_sketch_paths,
        test_photo_paths,
        eval_model_params.in_size,
        eval_model_params.pad_size,
        eval_model_params.out_size,
        eval_model_params.batch_size,
        eval_model_params.is_training
    )

    result = [
        train_set, test_set, train_model_params, eval_model_params
    ]
    return result

###保存tensorflow张量列表### 已弃用
def save_var_list(contain_keys, exclude_keys):
    assert type(contain_keys) == list or type(exclude_keys) == list
    variables = tf.global_variables()
    if len(contain_keys) == 0 and len(exclude_keys) == 0:
        return variables
    vars = []
    for var in variables:
        exclude_flag = False
        for key in exclude_keys:
            if var.name.find(key) >= 0:
                exclude_flag = True
                break
        if exclude_flag:
            break

        contain_flag = False
        for key in contain_keys:
            if var.name.find(key) >= 0:
                contain_flag = True
                break
        if contain_flag:
            vars.append(var)

    return vars

###加载模型，默认方法，即加载模型目录下最新的模型###
def load_checkpoint(sess, model_path, var_list):
    saver = tf.train.Saver(var_list=var_list)
    print('Loading model %s' % model_path)
    saver.restore(sess, model_path)

###加载模型，根据模型文件的名称记载模型###
def load_checkpoint2(sess, checkpoint_path, var_list):
    saver = tf.train.Saver(var_list)
    ckpt = tf.train.get_checkpoint_state(checkpoint_path)
    print('Loading model %s' % ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)

###保存模型###
def save_model(sess, saver, checkpoint_path, global_step):
    print('saving model %s.' % checkpoint_path)
    print('global_step %i.' % global_step)
    saver.save(sess, checkpoint_path, global_step=global_step)
    return checkpoint_path + "-{}".format(global_step)

###tensorflow保存日志方法###
def create_summary(summary_writer, summ_map, step):
    for summ_key in summ_map:
        summ_value = summ_map[summ_key]
        summ = tf.summary.Summary()
        summ.value.add(tag=summ_key, simple_value=float(summ_value))
        summary_writer.add_summary(summ, step)
    summary_writer.flush()

###模型测试方法###
def evaluate_model(sess, model, test_set, save_path):
    length = test_set.sample_len
    for i in range(length):
        x1, x1_paths, x2, x2_paths, y, y_paths = test_set.get_batch(i)
        x1 = np.expand_dims(x1, 3)
        feed = {
            model.input_photo: y,
            model.input_sketch: x1
        }
        fake_y = sess.run(model.fake_y, feed)
        fake_y = np.clip(np.squeeze(fake_y), -1.0, 1.0)
        fake_y = fake_y[3:253, 28:228]
        fake_y = (fake_y + 1) * 127.5
        fake_y = np.rint(fake_y)

        path_split = y_paths[0].split("/")
        im_name = path_split[-3] + path_split[-1].replace("png", "jpg")
        #     utils.draw_image([np.squeeze(y)[3:253, 28:228], np.squeeze(x1)[3:253, 28:228], fake_y], (250, 200),
        #                      "/data/shengqingjie/outputs/Fss/snapshot/{}/best_model".format(FLAGS.sub_node))
        im = Image.fromarray(np.uint8(fake_y))
        im.save(os.path.join(save_path, im_name))

###模型训练方法###
def train(sess, train_model, eval_model, train_set, valid_set):
    # Setup summary writer.
    train_params = train_model.hps
    summary_writer = tf.summary.FileWriter(train_params.log_root)

    print('-' * 100)

    # Calculate trainable params.
    t_vars = tf.trainable_variables()
    count_t_vars = 0
    ###网络参数保存的文件路径###
    model_network_path = os.path.join(train_params.snapshot_root, "model_network.txt")
    f = open(model_network_path, "w")
    f.truncate()
    f.close()
    for var in t_vars:
        num_param = np.prod(var.get_shape().as_list())
        count_t_vars += num_param
        log = '%s | shape: %s | num_param: %i' % (var.name, str(var.get_shape()), num_param)
        print(log)
        ###保存网络参数###
        write_log(log, model_network_path)
    print('Total trainable variables %i.' % count_t_vars)
    print('-' * 100)

    # main train loop

    hps = train_model.hps
    model_save_path = os.path.join(train_params.snapshot_root, hps.data_type)
    ###最优模型保存路径的前缀###
    best_model_prefix = train_params.snapshot_root + "/best_model/fss_model"
    # create saver
    var_list = save_var_list(["fss"], ["vgg"])
    ###设置模型目录下最多保存的模型数量，会自动删除旧模型###
    saver = tf.train.Saver(var_list, max_to_keep=5)

    start = time.time()
    num_steps = hps.num_epochs * hps.steps_per_epoch
    step = sess.run(train_model.global_step)
    ###训练时用于测量FID指标时，需要用到的临时文件目录###
    path1 = "/data/shengqingjie/outputs/Fss/Results/Tmp/{}".format(hps.sub_node)
    os.makedirs(path1, exist_ok=True)
    ###测试时的ground-truth路径
    path2 = "/data/shengqingjie/outputs/Fss/Results/{}/GR/Sketch".format(hps.data_type.upper())
    ###测量模型初始化时的FID参数###
    fid_eval = FID.Evaluator(path1, path2, gpu=0)
    evaluate_model(sess, eval_model, valid_set, path1)
    best_fid = fid_eval()
    fid_summ = tf.summary.Summary()
    fid_summ.value.add(
        tag='FID', simple_value=best_fid)
    summary_writer.add_summary(fid_summ, 0)
    summary_writer.flush()
    # for _ in range(num_steps):
    while step <= num_steps:
        ###加载batch数据，x1是sketch，y是photo，x2是不匹配的sketch(研究过程中用到的，现已废弃)###
        x1, x1_paths, x2, x2_paths, y, y_paths = train_set.random_batch()
        ###读取的sketch数据是[256, 256]的，网络需要的输入是[256, 256, 1]###
        x1 = np.expand_dims(x1, 3)
        feed = {
            train_model.input_sketch: x1,
            train_model.input_photo: y
        }

        # _, loss_Dy = sess.run([
        #     train_model.Dy_optimizer,
        #     train_model.loss_Dy
        # ], feed)
        ###判别器损失优化step###
        _, _, loss_Dx, loss_Dy = sess.run([
            train_model.Dx_optimizer,
            train_model.Dy_optimizer,
            train_model.loss_Dx,
            train_model.loss_Dy,
        ], feed)

        # _, _, loss_Dy1, loss_Dy2 = sess.run([
        #     train_model.Dy1_optimizer,
        #     train_model.Dy2_optimizer,
        #     train_model.loss_Dy1,
        #     train_model.loss_Dy2
        # ], feed)
        ###F-Lmser优化step###
        _, loss_F = sess.run([
            train_model.F_optimizer,
            train_model.loss_F
        ], feed)
        ###G-Lmser优化step###
        _, _, loss_G, step = sess.run([
            train_model.G_optimizer,
            train_model.step_op,
            train_model.loss_G,
            train_model.global_step
        ], feed)

        # _, _, _, loss_G, loss_F, step = sess.run([
        #     train_model.G_optimizer,
        #     train_model.F_optimizer,
        #     train_model.step_op,
        #     train_model.loss_G,
        #     train_model.loss_F,
        #     train_model.global_step
        # ], feed)

        # _, _, loss_G, step = sess.run([
        #     train_model.G_optimizer,
        #     train_model.step_op,
        #     train_model.loss_G,
        #     train_model.global_step
        # ], feed)
        # w_f = w_f[0]
        ###每20步打印一次当前的损失函数值###
        if step % 20 == 0 and step > 0:
            end = time.time()
            time_taken = end - start
            output_format = ('step: %d, '
                             'loss_Dx: %.4f, '
                             'loss_Dy: %.4f, '
                             'loss_G: %.4f, '
                             'loss_F: %.4f, '
                             'train_time_taken: %.4f.'
                             )
            output_values = (step,
                             loss_Dx,
                             loss_Dy,
                             loss_G,
                             loss_F,
                             time_taken
                             )
            output_log = output_format % output_values
            # write_log(output_log)
            print(output_log)
            start = time.time()

        epoch = int(step / hps.steps_per_epoch)
        # if True:
        ###每一个epoch保存并测试一次模型###
        if step > 0 and step % hps.steps_per_epoch == 0:
        # if epoch >= 250 and epoch % 50 == 0 and step % hps.steps_per_epoch == 0:
            model_name = save_model(sess, saver, model_save_path, step)
            # if epoch > hps.num_epochs - 100:
            if True:
                evaluate_model(sess, eval_model, valid_set, path1)
                fid = fid_eval()
                #
                fid_summ = tf.summary.Summary()
                fid_summ.value.add(
                    tag='FID', simple_value=fid)
                summary_writer.add_summary(fid_summ, epoch)
                summary_writer.flush()

                end = time.time()
                eval_time_taken = end - start
                start = time.time()

                output_format = ('eval model finished, FID={}, best_FID={}, '
                                 'eval_time_taken: %.4f'.format(fid, best_fid))
                output_values = eval_time_taken
                output_log = output_format % output_values

                # write_log(output_log)
                print(output_log)

                ###保存最优模型###
                if fid < best_fid:
                    best_fid = fid
                    # if epoch > hps.num_epochs - 100:
                    os.system("cp {}.index {}.index".format(model_name, best_model_prefix))
                    os.system("cp {}.meta {}.meta".format(model_name, best_model_prefix))
                    os.system("cp {}.data-00000-of-00001 {}.data-00000-of-00001".format(model_name, best_model_prefix))


def main():
    """Load model params, save config file and start training."""
    ###初始化参数###
    params = option.init()
    ###初始化显卡使用id号###
    os.environ['CUDA_VISIBLE_DEVICES'] = params.gpu
    params.log_root = params.log_root + params.sub_node
    params.snapshot_root = params.snapshot_root + params.sub_node

    np.set_printoptions(precision=8, edgeitems=6, linewidth=200, suppress=True)

    print('Loading data files.')
    print('-' * 100)
    ###数据集加载###
    datasets = load_dataset(params)

    train_set = datasets[0]
    valid_set = datasets[1]
    train_model_params = datasets[2]
    eval_model_params = datasets[3]

    print('Hyperparams:')
    for key, val in vars(train_model_params).items():
        print('%s = %s' % (key, str(val)))

    reset_graph()
    ###建立训练模型###
    train_model = fss_model.Model(train_model_params)
    ###建立测试模型，与训练模型共享网络权重参数，部分参数为测试时用到的参数###
    eval_model = fss_model.Model(eval_model_params)

    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True
    tfconfig.allow_soft_placement = True
    tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.95
    sess = tf.InteractiveSession(config=tfconfig)
    ###初始化tensorflow张量###
    sess.run(tf.global_variables_initializer())
    ###是否重启训练，默认从上次保存的最优模型开始训练###
    if params.resume_training:
        var_list = save_var_list(["fss"], [])
        # load_checkpoint(sess, params.snapshot_root + "/cufsf-5750", var_list)
        # load_checkpoint(sess, "/data/shengqingjie/outputs/Fss/snapshot/mycGAN_cufsf_byy_pretrain/cufsf-50000", var_list)
        load_checkpoint(sess, params.snapshot_root + "/best_model/fss_model", var_list)

    os.makedirs(params.log_root, exist_ok=True)
    os.makedirs(params.snapshot_root, exist_ok=True)
    os.makedirs(params.snapshot_root + "/best_model", exist_ok=True)
    # Write config file to json file.
    with tf.gfile.Open(
            os.path.join(params.snapshot_root, 'model_config.json'), 'w') as f:
        json.dump(vars(train_model_params), f, indent=True)
    ###开始训练###
    train(sess, train_model, eval_model, train_set, valid_set)


if __name__ == '__main__':
    ###主函数入口###
    main()
