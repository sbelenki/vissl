from classy_vision.dataset.transforms import register_transform
from classy_vision.dataset.transforms.classy_transform import ClassyTransform
import numpy as np
from numpy.fft import fftshift, ifftshift, fftn


@register_transform("SpatialToFrequency")
class SpatialToFrequency(ClassyTransform):
    """
    Converts an image from the spatial domain into an image in the frequency domain
    """

    def __call__(self, img) -> np.ndarray:
        #dim = range(img.ndim)
        
        tt = np.array(img)
        
        dim=(0,1)
        
        k = fftshift(fftn(ifftshift(tt, axes=dim), s=None, axes=dim), axes=dim)
        k /= np.sqrt(np.prod(np.take(tt.shape, dim)))

        return k
