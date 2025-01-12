# AUTOGENERATED! DO NOT EDIT! File to edit: ../../../nbs/libs/models/load_empty_model.ipynb.

# %% auto 0
__all__ = ['load_empty_model']

# %% ../../../nbs/libs/models/load_empty_model.ipynb 2
from libs.models.segnet import SegNet
from libs.models.unet import UNet
from libs.models.deconvnet import DeconvNet


def load_empty_model(architecture, n_classes):
    models = {
        'segnet': SegNet,
        'unet': UNet,
        'deconvnet': DeconvNet
    }

    ModelClass = models[architecture]
    model = ModelClass(n_channels=1, n_classes=n_classes)

    return model
