from losses.loss import FRCNNLoss


def build_loss(opts):
    criterion = FRCNNLoss(opts)
    return criterion

