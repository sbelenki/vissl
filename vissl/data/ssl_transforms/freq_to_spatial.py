from typing import Any, Dict
from classy_vision.dataset.transforms import register_transform
from classy_vision.dataset.transforms.classy_transform import ClassyTransform
from PIL import Image
import numpy as np
import torch
import torch.fft


@register_transform("FrequencyToSpatial")
class FrequencyToSpatial(ClassyTransform):
    """
    Converts an image in the frequency domain into an image in the spatial domain
    
    Args:
        index (int): The index into the stack of MRI slices to pull the image from
        mode (str): The mode to transform the Image to.  For 3 dimensional images choose "RGB".
    """

    def __init__(self, index: int = 12, mode: str = "RGB"):
        self.index = index
        self.mode = mode
    
    #Credit: fastMRI: https://github.com/facebookresearch/fastMRI/blob/43718c3f1d0be52228c65d3e2b27ac492122c0d5/banding_removal/fastmri/data/transforms.py#L490
    def roll(self, x, shift, dim):
        """
        Similar to np.roll but applies to PyTorch Tensors
        """
        if isinstance(shift, (tuple, list)):
            assert len(shift) == len(dim)
            for s, d in zip(shift, dim):
                x = self.roll(x, s, d)
            return x
        shift = shift % x.size(dim)
        if shift == 0:
            return x
        left = x.narrow(dim=dim, start=0, length=x.size(dim) - shift)
        right = x.narrow(dim=dim, start=x.size(dim) - shift, length=shift)
        return torch.cat((right, left), dim=dim)
    
    #Credit: fastMRI: https://github.com/facebookresearch/fastMRI/blob/43718c3f1d0be52228c65d3e2b27ac492122c0d5/banding_removal/fastmri/data/transforms.py#L528
    def ifftshift(self, x, dim=None):
        """
        Similar to np.fft.ifftshift but applies to PyTorch Tensors
        """
        if dim is None:
            dim = tuple(range(x.dim()))
            shift = [(dim + 1) // 2 for dim in x.shape] #TODO: looks wrong
        elif isinstance(dim, int):
            shift = (x.shape[dim] + 1) // 2
        else:
            shift = [(x.shape[i] + 1) // 2 for i in dim]
        return self.roll(x, shift, dim)
    
    # Credit: fastMRI: https://github.com/facebookresearch/fastMRI/blob/43718c3f1d0be52228c65d3e2b27ac492122c0d5/banding_removal/fastmri/data/transforms.py#L506
    def fftshift(self, x, dim=None):
        """
        Similar to np.fft.fftshift but applies to PyTorch Tensors
        """
        if dim is None:
            dim = tuple(range(x.dim()))
            shift = [dim // 2 for dim in x.shape]
        elif isinstance(dim, int):
            shift = x.shape[dim] // 2
        else:
            shift = [x.shape[i] // 2 for i in dim]
        return self.roll(x, shift, dim)
    
    # Credit: fastMRI: https://github.com/facebookresearch/fastMRI/blob/43718c3f1d0be52228c65d3e2b27ac492122c0d5/fastmri/fftc.py#L39
    def ifft2c(self, data: torch.Tensor) -> torch.Tensor:
        """
        Apply centered 2-dimensional Inverse Fast Fourier Transform.

        Args:
            data: Complex valued input data containing at least 3 dimensions:
                dimensions -3 & -2 are spatial dimensions and dimension -1 has size
                2. All other dimensions are assumed to be batch dimensions.

        Returns:
            The IFFT of the input.
        """
        if not data.shape[-1] == 2:
            raise ValueError("Tensor does not have separate complex dim.")

        data = self.ifftshift(data, dim=[-3, -2])
        data = torch.view_as_real(
            torch.fft.ifftn(  # type: ignore
                torch.view_as_complex(data), dim=(-2, -1), norm="ortho"
            )
        )
        data = self.fftshift(data, dim=[-3, -2])

        return data
        
    #Credit: fastMRI: https://github.com/facebookresearch/fastMRI/blob/43718c3f1d0be52228c65d3e2b27ac492122c0d5/fastmri/math.py#L55
    def complex_abs(self, data: torch.Tensor) -> torch.Tensor:
        """
        Compute the absolute value of a complex valued input tensor.
        Args:
            data: A complex valued tensor, where the size of the final dimension
                should be 2.
        Returns:
            Absolute value of data.
        """
        if not data.shape[-1] == 2:
            raise ValueError("Tensor does not have separate complex dim.")

        return (data ** 2).sum(dim=-1).sqrt()

    #Credit: fastMRI: https://github.com/facebookresearch/fastMRI/blob/43718c3f1d0be52228c65d3e2b27ac492122c0d5/fastmri/data/transforms.py#L17
    def to_tensor(self, data: np.ndarray) -> torch.Tensor:
        """
        Convert numpy array to PyTorch tensor.
        For complex arrays, the real and imaginary parts are stacked along the last
        dimension.
        Args:
            data: Input numpy array.
        Returns:
            PyTorch version of data.
        """
        if np.iscomplexobj(data):
            data = np.stack((data.real, data.imag), axis=-1)

        return torch.from_numpy(data)

    def __call__(self, kspace_img) -> Image:
        kspace_img_t = self.to_tensor(kspace_img)  # Convert from numpy array to pytorch tensor
        slice_image = self.ifft2c(kspace_img_t)  # Apply Inverse Fourier Transform to get the complex image
        slice_image_abs = self.complex_abs(slice_image)  # Compute absolute value to get a real image

        tmp = slice_image_abs.numpy()
        tmp = (tmp * 255 / np.max(tmp)).astype('uint8')
        tmp = Image.fromarray(tmp)

        # need to make sure the image has 3 channels as that is what unet and other models expect
        img = Image.new(mode=self.mode, size=tmp.size)
        img.paste(tmp, box=[0, 0])

        return img

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FrequencyToSpatial":
        """
        Instantiates FrequencyToSpatial from configuration.

        Args:
            config (Dict): arguments for for the transform

        Returns:
            FrequencyToSpatial instance.
        """
        index = config.get("index", 12)
        assert index > 0, "Index must be greater than 0"

        mode = config.get("mode", "RGB")

        return cls(index=index, mode=mode)
