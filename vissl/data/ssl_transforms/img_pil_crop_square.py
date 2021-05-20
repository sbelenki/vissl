from typing import Any, Dict, List

from classy_vision.dataset.transforms import register_transform
from classy_vision.dataset.transforms.classy_transform import ClassyTransform
from PIL import Image, ImageOps

@register_transform("ImgPilCropSquare")
class ImgPilCropSquare(ClassyTransform):
    
    def __call__(self, img: Image):
        smallestdim = min(img.size[0], img.size[1])
        size = (smallestdim, smallestdim)
        return ImageOps.fit(img, size)
    
