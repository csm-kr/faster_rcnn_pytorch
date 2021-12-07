from model.rpn import RPN
from torchvision.models import vgg16_bn


def build_model(model_config):
    model_name = model_config['model_name']
    model = None
    assert model_name in ['rpn', 'faster_rcnn']
    if model_name == 'rpn':
        model = RPN(conv_layers=vgg16_bn(pretrained=True))
    return model