from loss.rpn_loss import RPNLoss
from loss.faster_rcnn_loss import FasterRCNNLoss


def build_loss(model_config, coder):
    model_name = model_config['model_name']
    assert model_name in ['rpn', 'faster_rcnn']
    if model_name == 'rpn':
        loss = RPNLoss(coder)
    else:
        loss = FasterRCNNLoss()
    return loss