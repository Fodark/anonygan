import torch

from .arcface.backbones import get_model


def ArcFace(arcface_arch):
    model = get_model(f"{arcface_arch}", fp16=False)
    model.load_state_dict(torch.load(f"src/models/arcface/backbone_{arcface_arch}.pth"))

    return model.eval()
