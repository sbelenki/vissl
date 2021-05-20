from classy_vision.dataset.transforms import register_transform
from classy_vision.dataset.transforms.classy_transform import ClassyTransform
import numpy as np
from typing import Optional, Sequence, Tuple, Union, Dict, Any
import torch
import contextlib

# Credit: fastMRI: https://github.com/facebookresearch/fastMRI/blob/43718c3f1d0be52228c65d3e2b27ac492122c0d5/fastmri/data/subsample.py#L15
@contextlib.contextmanager
def temp_seed(rng: np.random, seed: Optional[Union[int, Tuple[int, ...]]]):
    if seed is None:
        try:
            yield
        finally:
            pass
    else:
        state = rng.get_state()
        rng.seed(seed)
        try:
            yield
        finally:
            rng.set_state(state)

# Credit: fastMRI: https://github.com/facebookresearch/fastMRI/blob/43718c3f1d0be52228c65d3e2b27ac492122c0d5/fastmri/data/subsample.py#L31
class MaskFunc:
    """
    An object for GRAPPA-style sampling masks.
    This crates a sampling mask that densely samples the center while
    subsampling outer k-space regions based on the undersampling factor.
    """

    def __init__(self, center_fractions: Sequence[float], accelerations: Sequence[int]):
        """
        Args:
            center_fractions: Fraction of low-frequency columns to be retained.
                If multiple values are provided, then one of these numbers is
                chosen uniformly each time.
            accelerations: Amount of under-sampling. This should have the same
                length as center_fractions. If multiple values are provided,
                then one of these is chosen uniformly each time.
        """
        if not len(center_fractions) == len(accelerations):
            raise ValueError(
                "Number of center fractions should match number of accelerations"
            )

        self.center_fractions = center_fractions
        self.accelerations = accelerations
        self.rng = np.random.RandomState()  # pylint: disable=no-member

    def __call__(
        self, shape: Sequence[int], seed: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> torch.Tensor:
        raise NotImplementedError

    def choose_acceleration(self):
        """Choose acceleration based on class parameters."""
        choice = self.rng.randint(0, len(self.accelerations))
        center_fraction = self.center_fractions[choice]
        acceleration = self.accelerations[choice]

        return center_fraction, acceleration

# Credit: fastMRI: https://github.com/facebookresearch/fastMRI/blob/43718c3f1d0be52228c65d3e2b27ac492122c0d5/fastmri/data/subsample.py#L72
class RandomMaskFunc(MaskFunc):
    """
    RandomMaskFunc creates a sub-sampling mask of a given shape.
    The mask selects a subset of columns from the input k-space data. If the
    k-space data has N columns, the mask picks out:
        1. N_low_freqs = (N * center_fraction) columns in the center
           corresponding to low-frequencies.
        2. The other columns are selected uniformly at random with a
        probability equal to: prob = (N / acceleration - N_low_freqs) /
        (N - N_low_freqs). This ensures that the expected number of columns
        selected is equal to (N / acceleration).
    It is possible to use multiple center_fractions and accelerations, in which
    case one possible (center_fraction, acceleration) is chosen uniformly at
    random each time the RandomMaskFunc object is called.
    For example, if accelerations = [4, 8] and center_fractions = [0.08, 0.04],
    then there is a 50% probability that 4-fold acceleration with 8% center
    fraction is selected and a 50% probability that 8-fold acceleration with 4%
    center fraction is selected.
    """

    def __call__(
        self, shape: Sequence[int], seed: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> torch.Tensor:
        """
        Create the mask.
        Args:
            shape: The shape of the mask to be created. The shape should have
                at least 3 dimensions. Samples are drawn along the second last
                dimension.
            seed: Seed for the random number generator. Setting the seed
                ensures the same mask is generated each time for the same
                shape. The random state is reset afterwards.
        Returns:
            A mask of the specified shape.
        """
        if len(shape) < 3:
            raise ValueError("Shape should have 3 or more dimensions")

        with temp_seed(self.rng, seed):
            num_cols = shape[-2]
            center_fraction, acceleration = self.choose_acceleration()

            # create the mask
            num_low_freqs = int(round(num_cols * center_fraction))
            prob = (num_cols / acceleration - num_low_freqs) / (
                num_cols - num_low_freqs
            )
            mask = self.rng.uniform(size=num_cols) < prob
            pad = (num_cols - num_low_freqs + 1) // 2
            mask[pad : pad + num_low_freqs] = True

            # reshape the mask
            mask_shape = [1 for _ in shape]
            mask_shape[-2] = num_cols
            mask = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))

        return mask

# Credit: fastMRI: https://github.com/facebookresearch/fastMRI/blob/43718c3f1d0be52228c65d3e2b27ac492122c0d5/fastmri/data/subsample.py#L136
class EquispacedMaskFunc(MaskFunc):
    """
    EquispacedMaskFunc creates a sub-sampling mask of a given shape.
    The mask selects a subset of columns from the input k-space data. If the
    k-space data has N columns, the mask picks out:
        1. N_low_freqs = (N * center_fraction) columns in the center
           corresponding tovlow-frequencies.
        2. The other columns are selected with equal spacing at a proportion
           that reaches the desired acceleration rate taking into consideration
           the number of low frequencies. This ensures that the expected number
           of columns selected is equal to (N / acceleration)
    It is possible to use multiple center_fractions and accelerations, in which
    case one possible (center_fraction, acceleration) is chosen uniformly at
    random each time the EquispacedMaskFunc object is called.
    Note that this function may not give equispaced samples (documented in
    https://github.com/facebookresearch/fastMRI/issues/54), which will require
    modifications to standard GRAPPA approaches. Nonetheless, this aspect of
    the function has been preserved to match the public multicoil data.
    """

    def __call__(
        self, shape: Sequence[int], seed: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> torch.Tensor:
        """
        Args:
            shape: The shape of the mask to be created. The shape should have
                at least 3 dimensions. Samples are drawn along the second last
                dimension.
            seed: Seed for the random number generator. Setting the seed
                ensures the same mask is generated each time for the same
                shape. The random state is reset afterwards.
        Returns:
            A mask of the specified shape.
        """
        if len(shape) < 3:
            raise ValueError("Shape should have 3 or more dimensions")

        with temp_seed(self.rng, seed):
            center_fraction, acceleration = self.choose_acceleration()
            num_cols = shape[-2]
            num_low_freqs = int(round(num_cols * center_fraction))

            # create the mask
            mask = np.zeros(num_cols, dtype=np.float32)
            pad = (num_cols - num_low_freqs + 1) // 2
            mask[pad : pad + num_low_freqs] = True

            # determine acceleration rate by adjusting for the number of low frequencies
            adjusted_accel = (acceleration * (num_low_freqs - num_cols)) / (
                num_low_freqs * acceleration - num_cols
            )
            offset = self.rng.randint(0, round(adjusted_accel))

            accel_samples = np.arange(offset, num_cols - 1, adjusted_accel)
            accel_samples = np.around(accel_samples).astype(np.uint)
            mask[accel_samples] = True

            # reshape the mask
            mask_shape = [1 for _ in shape]
            mask_shape[-2] = num_cols
            mask = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))

        return mask


@register_transform("ApplyFrequencyMask")
class ApplyFrequencyMask(ClassyTransform):
    
    def __init__(self, mask_type:str, center_fractions: Sequence[float], accelerations: Sequence[int],):
        
        # Credit: fastMRI: https://github.com/facebookresearch/fastMRI/blob/43718c3f1d0be52228c65d3e2b27ac492122c0d5/fastmri/data/subsample.py#L205
        if mask_type == "random":
            self.mask_func = RandomMaskFunc(center_fractions, accelerations)
        elif mask_type == "equispaced":
            self.mask_func = EquispacedMaskFunc(center_fractions, accelerations)
        else:
            raise Exception(f"{mask_type} not supported")
        
    def __call__(self, data:np.ndarray) -> np.ndarray:
        #print(f"data shape {data.shape}, {data.dtype}")
        # Credit: https://github.com/facebookresearch/fastMRI/blob/43718c3f1d0be52228c65d3e2b27ac492122c0d5/banding_removal/fastmri/data/transforms.py#L66
        shape = data.shape
        shape += (1,)
        #shape[:-3] = 1
        #print(f"shape shape {shape}")
        mask = self.mask_func(shape, None)
        #mask = mask.to(data.device)

        masked_data = data * mask.numpy() + 0.0 # The + 0.0 removes the sign of the zeros
        masked_data = masked_data[0,:,:]
        #print(f"masked data {masked_data.shape}")
        return masked_data #, mask, num_low_frequencies
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ApplyFrequencyMask":
        """
        Instantiates ApplyFrequencyMask from configuration.

        Args:
            config (Dict): arguments for for the transform

        Returns:
            ApplyFrequencyMask instance.
        """
        maskType = config.get("mask_type", "random")
        centerFractions = config.get("center_fractions", [0.04])
        accelerations = config.get("accelerations", [8])

        return cls(mask_type=maskType, center_fractions=centerFractions, accelerations=accelerations)