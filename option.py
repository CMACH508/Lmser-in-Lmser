import argparse


def init():
    parser = argparse.ArgumentParser(description='Parameters ')
    parser.add_argument('--data_dir', type=str, help='The directory of sketch data of the dataset.',
                        default='/data/shengqingjie/datasets/FaceSketch')
    parser.add_argument('--log_root', type=str, help='Directory to store print log.',
                        default='/data/shengqingjie/outputs/Fss/log/')
    parser.add_argument('--data_type', type=str, help='cufs or cufsf',
                        default='cufs')
    parser.add_argument('--snapshot_root', type=str, help='Directory to store model checkpoints.',
                        default='/data/shengqingjie/outputs/Fss/snapshot/')
    parser.add_argument('--resume_training', type=bool, help='Set to true to load previous checkpoint.',
                        default=False)
    parser.add_argument('--data_load_mode', type=str, help='Load or create database split lists.',
                        default='create')
    parser.add_argument('--sub_node', type=str, help='Sub node for saving models and results.',
                        default='Lmserinlmser_cufs')
    parser.add_argument('--gpu', type=str, help='Gpu device for training or testing.',
                        default='1')
    parser.add_argument('--pool_size', type=int, help='Pool of fake image in GAN model.',
                        default=50)
    parser.add_argument('--in_size', type=tuple, help='Image size when reading. The original image will be resized in this size.',
                        default=(200, 250))
    parser.add_argument('--pad_size', type=tuple, help='Image size when precessing. The image read will be padded in this size for cropping.',
                        default=(286, 286))
    parser.add_argument('--out_size', type=tuple, help='Image size when precessed. The image size which fed into model.',
                        default=(256, 256))
    parser.add_argument('--lr', type=float, help='Learning rate.',
                        default=0.0001)
    parser.add_argument('--min_learning_rate', type=float, help='Minimum learning rate.',
                        default=0.0000001)
    parser.add_argument('--num_epochs', type=int, help='Total number of epochs of training. Keep large.',
                        default=100)
    parser.add_argument('--batch_size', type=int, help='Training batch size.',
                        default=1)
    parser.add_argument('--steps_per_epoch', type=int, help='Training steps per epoch. Assigned when loading dataset during training.',
                        default=0)
    parser.add_argument('--start_decay_epochs', type=int, help='Epoch when learning rate start to decay.',
                        default=100)
    parser.add_argument('--decay_epochs', type=int, help='Learning rate decay epoch periods.',
                        default=400)
    parser.add_argument('--alpha', type=float, help='alpha',
                        default=0.5)
    parser.add_argument('--beta1', type=float, help='beta1 for Adam optimizer',
                        default=0.9)
    parser.add_argument('--beta2', type=float, help='beta2 for Adam optimizer',
                        default=0.999)
    parser.add_argument('--moving_decay', type=float, help='moving average decay for generator',
                        default=0.9999)
    parser.add_argument('--lambda1', type=float, help='Weight of the photo cyclic loss.',
                        default=10)
    parser.add_argument('--lambda2', type=float, help='Weight of the sketch cyclic loss.',
                        default=10)
    parser.add_argument('--l1_lambda', type=float, help='Weight of l1 loss in cGAN.',
                        default=10)
    parser.add_argument('--ld', type=float, help='The gradient penalty lambda',
                        default=10.0)
    parser.add_argument('--gan_type', type=str, help='[gan / lsgan / wgan-gp / wgan-lp / dragan / hinge]',
                        default='lsgan')
    parser.add_argument('--share', type=bool, help='Is model sharing weight? Recommend keeping true.',
                        default=False)
    parser.add_argument('--is_training', type=bool, help='Is model training? Recommend keeping true.',
                        default=True)
    parser.add_argument('--separate', type=bool, help='Is model training? Recommend keeping true.',
                        default=False)

    opt = parser.parse_args()
    return opt