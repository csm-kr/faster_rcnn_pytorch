import torch
# from models.model import FRCNN
from models.new_model import FRCNN
from torch.nn.parallel import DistributedDataParallel as DDP


def build_model(opts):
    if opts.distributed:
        model = FRCNN(num_classes=opts.num_classes)
        model = model.cuda(int(opts.gpu_ids[opts.rank]))
        model = DDP(module=model,
                    device_ids=[int(opts.gpu_ids[opts.rank])],
                    find_unused_parameters=False)
    else:
        # IF DP
        model = FRCNN(num_classes=opts.num_classes)
        model = torch.nn.DataParallel(module=model, device_ids=[int(id) for id in opts.gpu_ids])
    return model

