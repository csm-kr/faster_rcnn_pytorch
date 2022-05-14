from loss.rpn_loss import RPNLoss
from loss.faster_rcnn_loss import FRCNNLoss


def build_loss(model_config, coder):
    model_name = model_config['model_name']
    assert model_name in ['rpn', 'frcnn']
    if model_name == 'rpn':
        loss = RPNLoss(coder)
    elif model_name == 'frcnn':
        loss = FRCNNLoss(coder)
    return loss