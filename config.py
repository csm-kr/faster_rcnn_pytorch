import argparse


def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)

    # visualization
    parser.add_argument('--visdom_port', type=int, default=8098)
    parser.add_argument('--vis_step', type=int, default=100)
    parser.add_argument('--save_step', type=int, default=50000)

    # save
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--log_dir', type=str, default='./logs')

    # dataset
    parser.add_argument('--name', type=str, default='faster_rcnn', help='experiment name')   # FIXME
    # parser.add_argument('--root', type=str, default=r'/home/cvmlserver7/Sungmin/data/voc')
    parser.add_argument('--root', type=str, default=r'D:\data\voc')
    parser.add_argument('--data_type', type=str, default='voc')

    # training
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument('--epoch', type=int, default=14)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--warmup_epoch', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)

    # testing
    parser.add_argument('--seed', type=int, default=0)
    parser.set_defaults(is_test=False)
    parser.add_argument('--testing', dest='is_test', action='store_true')
    parser.set_defaults(visualization=False)
    parser.set_defaults(test_vis=False)
    parser.add_argument('--test_epoch', type=str, default='best')
    # parser.add_argument('--visualization', dest='is_test', action='store_true')

    # demo
    # parser.add_argument('--demo_root', type=str, help='set demo root')
    parser.add_argument('--demo_root', type=str, default=r'C:\Users\csm81\Desktop\yesul', help='set demo root')
    parser.add_argument('--demo_epoch', type=str, default='best')
    parser.add_argument('--demo_image_type', type=str, default='jpg')
    parser.set_defaults(demo_vis=True)

    # for multi-gpu
    parser.add_argument('--gpu_ids', nargs="+", default=['0'])   # usage : --gpu_ids 0, 1, 2, 3
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)

    return parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser('faster rcnn training', parents=[get_args_parser()])
    opts = parser.parse_args()

    opts.world_size = len(opts.gpu_ids)
    opts.num_workers = len(opts.gpu_ids) * 4
    print(opts)
