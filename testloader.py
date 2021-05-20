from vissl.data.fastmri_dataset import FastMRIDataSet
from vissl.config import AttrDict
from vissl.data.ssl_transforms.freq_to_spatial import FrequencyToSpatial
from vissl.data.ssl_transforms.spatial_to_freq import SpatialToFrequency
from vissl.data.ssl_transforms.freq_apply_mask import ApplyFrequencyMask
# from vissl.data.ssl_transforms.rgb_to_grayscale import RGBToGrayScale
from PIL import Image

attributes = AttrDict({"DATA": {"INDEX": 18}})

data = FastMRIDataSet(cfg=attributes,path="/mnt/d/data", split="train")
#data = FastMRIDataSet(cfg=attributes, path="/Users/ylichman/classes/dl/final/data", split="train")
# print(data.num_samples())

spatial_image, _ = data[18]

# tmp = ( tmp * 255 / np.max(tmp)).astype('uint8')

# onedimImage = Image.fromarray(tmp)
# imaget = Image.fromarray(spatial_image[:,:,0], )
# imaget.save("test_spatial.png")

# gray = RGBToGrayScale()(spatial_image)

freq_image = SpatialToFrequency()(spatial_image)  # this needs to be a tensor

print(freq_image.shape)

print(f'Processes spatial to freq with shape: {freq_image.shape}, {freq_image[0, 0].dtype}')
spatial_image = FrequencyToSpatial()(freq_image)
spatial_image.save("test_frequency_to_spacial.png")

masked_image = ApplyFrequencyMask(mask_type="random",center_fractions=[0.2], accelerations=[8])(freq_image)
adj_spatial_iamge = FrequencyToSpatial()(masked_image)
adj_spatial_iamge.save("test_frequency_adjusted_to_spacial.png")