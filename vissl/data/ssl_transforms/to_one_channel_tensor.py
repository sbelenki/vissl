from classy_vision.dataset.transforms import register_transform
from classy_vision.dataset.transforms.classy_transform import ClassyTransform
import numpy as np

@register_transform("ToOneChannelTensor")
class ToOneChannelTensor(ClassyTransform):
    """
    Converts an image from 3-channle to 1-channel
    """

    def __call__(self, img) -> np.ndarray:
        return img[0].unsqueeze(0)
