项目目录情况：
Lmser_in_Lmser......
    ......data                       ###数据集划分txt文件存放目录
    ......model                      ###模型实现文件存放目录
    ......NLDA                       ###matlab用来测试NLDA指标的文件目录，需要单独在服务器上用matlab命令调用，事先需要设置好测试的方法
    ......pytorch_fid                ###FID指标pytorch实现文件存放目录，不需要单独运行，调用就可以
    ......util                       ###数据预处理实现方法所在目录
    FeatureSIM.m                     ###测试FSIM指标实现方法文件，不需要单独运行，test_Method.py调用
    option.py                        ###环境参数文件###
    test_Method.py                   ###测试FID、FSIM指标方法
    test_lmser_in_lmser.py           ###模型测试文件，根据测试集人脸照片生成素描图片，用于test_Method.py测试具体指标
    train_lmser_in_lmser.py          ###模型训练文件


相关命令举例：
训练：
    source activate py36             ###激活conda中的py36环境
    tmux 或 tmux a                   ###打开tmux终端复用器用于后台训练模型
    python train_lmser_in_lmser.py --gpu "0,1" --data_type CUFS --sub_node lmserinlmser_cufs
                                     ###运行训练模型进程，例子中主要设置了gpu，数据集类型，模型保存路径(即测试时的方法名)
                                     ###具体可见option.py文件
    python train_lmser_in_lmser.py --gpu "0,1" --data_type CUFS --sub_node lmserinlmser_cufs --resume_training 1
                                     ###重启训练进程，由于API问题，resume_training等bool类型参数需要设置默认值为false
                                     ###在需要设置为true时，只要用--resume_training + 随便一个值就可以
测试：
    python test_lmser_in_lmser.py --gpu XX --sub_node XX --data_type XX
                                     ###只有这三种参数，与option.py没有关系。通常本机IDE运行即可，不需要在服务器上用命令行运行
    python test_Method.py            ###本机IDE运行
    /data/shengqingjie/matlab/bin/matlab -nodisplay -r NLDA_multiple.m
    /data/shengqingjie/matlab/bin/matlab -nodisplay -r NLDA_single.m
                                     ###测试NLDA指标，需要先cd到NLDA_multiple.m、NLDA_single.m所在目录，不能在tmux终端运行

