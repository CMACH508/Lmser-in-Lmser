import numpy as np
import os
from PIL import Image
import matlab.engine
from pytorch_fid import fid_score as FID

data_dir = "/data/shengqingjie/outputs/Fss/Results/"
###数据集###
data_type = 'edges2shoes'
###方法，就是模型训练时所设置的sub_node参数，具体可查看模型保存的路径###
method = "pix2pix"
method_gr = "GR"
gpu = '3'
mode = "Sketch" # test Sketch or Photo

os.environ['CUDA_VISIBLE_DEVICES'] = gpu

def evaluate_model(img_path, tar_path):
    imgs = os.listdir(img_path)
    imgs.sort()
    tars = os.listdir(tar_path)
    tars.sort()

    length = len(imgs)
    eng = matlab.engine.start_matlab()
    fsim = 0
    for i in range(length):
        assert imgs[i] == tars[i]
        if mode == "Sketch":
            fake = Image.open(os.path.join(img_path, imgs[i])).convert("L")
            fake = np.array(fake, dtype=np.uint8)
            real = Image.open(os.path.join(tar_path, tars[i])).convert("L")
            real = np.array(real, dtype=np.uint8)
        else:
            fake = Image.open(os.path.join(img_path, imgs[i])).convert("RGB")
            fake = np.array(fake, dtype=np.uint8)
            real = Image.open(os.path.join(tar_path, tars[i])).convert("RGB")
            real = np.array(real, dtype=np.uint8)
        ###测试FSIM指标###
        fsim_return = eng.FeatureSIM(matlab.double(real.tolist()), matlab.double(fake.tolist()), nargout=2)
        fsim += fsim_return[1]

    fsim = fsim / length
    return fsim

img_path = data_dir + data_type + "/" + method + "/" + mode
tar_path = data_dir + data_type + "/" + method_gr + "/" + mode
# fsim = evaluate_model(img_path, tar_path)
# print('FSIM: %.4f' % fsim)
path1 = "/data/shengqingjie/outputs/Fss/Results/{}/{}/Sketch".format(data_type.upper(), method)
if not os.path.exists(path1):
    path1 = "/data/shengqingjie/outputs/Fss/Results/{}/{}".format(data_type.lower(), method)
path2 = "/data/shengqingjie/outputs/Fss/Results/{}/{}/Sketch".format(data_type.upper(), method_gr)
if not os.path.exists(path2):
    path2 = "/data/shengqingjie/outputs/Fss/Results/{}/{}".format(data_type.lower(), method_gr)
###测试FID指标###
fid_eval = FID.Evaluator(path1, path2, gpu=0)
fid = fid_eval()
print('FID: %.4f' % fid)
