import configargparse


def get_args_parser():
    # config
    parser = configargparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', is_config_file=True, help='config file path')       # TODO
    parser.add_argument('--name', type=str)                                             # TODO

    # visdom
    parser.add_argument('--visdom_port', type=int)                                      # TODO
    parser.add_argument('--vis_step', type=int, default=100)

    # data
    parser.add_argument('--data_root', type=str)                                        # TODO
    parser.add_argument('--data_type', type=str)                                        # TODO
    parser.add_argument('--num_classes', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=0)

    # training
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epoch', type=int, default=13)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument('--warmup_epoch', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--log_dir', type=str, default='./logs')

    # testing
    parser.add_argument('--test_epoch', type=str, default='best')
    parser.add_argument('--thres', type=float, default=0.05, help='score threshold - 0.05 for test 0.5 for demo')

    # demo
    parser.add_argument('--demo_epoch', type=str, default='best')
    parser.add_argument('--demo_root', type=str, help='set demo root')                 # TODO
    parser.add_argument('--demo_image_type', type=str, default='jpg')
    parser.set_defaults(demo_vis=False)
    parser.add_argument('--demo_vis_true', dest='demo_vis', action='store_true')

    # distributed
    parser.add_argument('--distributed_true', dest='distributed', action='store_true')
    parser.add_argument('--gpu_ids', nargs="+", default=['0'])   # usage : --gpu_ids 0, 1, 2, 3
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)

    return parser

