import argparse


class NYU_Options():
    def __init__(self):
        pass

    def initialize(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        # base configs
        parser.add_argument('--gpu_or_cpu',   type=str, default='gpu')
        parser.add_argument('--data_path', type=str, default='../Datasets/nyu_depth_v2/')
        parser.add_argument('--dataset',      type=str, default='nyudepthv2',
                            choices=['nyudepthv2', 'kitti', 'imagepath'])
        parser.add_argument('--exp_name',     type=str, default='test')

        # depth configs
        parser.add_argument('--max_depth',      type=float, default=10.0) 
        parser.add_argument('--max_depth_eval', type=float, default=10.0)
        parser.add_argument('--min_depth_eval', type=float, default=1e-3)        
        parser.add_argument('--do_kb_crop', type=int, default=0)
        parser.add_argument('--garg_crop', default=False)
        parser.add_argument('--kitti_crop', type=str, default='eigen_crop')
        parser.add_argument('--eigen_crop', default=True)

        parser.add_argument('--name', default='CHFNet')
        parser.add_argument('--root', default='.', help='Root folder to save data in')
        parser.add_argument('--random_seed', default=1)

        # experiment configs
        parser.add_argument('--epochs', type=int, default=50)

        parser.add_argument('--batch_size',   type=int, default=6)
        parser.add_argument('--workers',      type=int, default=24)
        parser.add_argument('--lr', type=float, default=1e-4)

        parser.add_argument('--crop_h', type=int, default=448)
        parser.add_argument('--crop_w', type=int, default=576)
        parser.add_argument('--log_dir', type=str, default='./logs/NYU')

        parser.add_argument('--validate_every', default=2000)
        parser.add_argument('--log_freq', default=200)
        parser.add_argument('--log_directory', default='train')
        return parser
