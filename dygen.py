

from dg_lib import DYImage
from tqdm import tqdm
import random
import os

init_file       = "ref" # just the root of the file name. .png will be added at the end
version         = 1     # number that gets appended to the file name, so you don't override if you're experimenting

scale_image     = 1     # makes image that much larger (multiplier)
color_count     = 64    # total number of colors used in painting
cluster         = 25    # Radius of paint blobs. Should be odd number! 
pixel_step      = 2     # how pixilated should the image be. Bigger number - larger pixels

glow_amount = (0.3, 0.3, 0.3) # amount of glow in each channel (R, G, B) should be between 0.0 and 1.0




# These are advanced settings
root_location = os.path.dirname(__file__)
number_of_masks = 2 # if you put more masks in the ref folder - bump up this number
k = 1
msk_m = pixel_step
use_map = False 
color_offset = 0
preglow_offset = 0
k *= scale_image




print(f"Starting {init_file}")
random.seed(2)
source_image = DYImage(f"{root_location}/ref/{init_file}.png").scaled( scale_image )
source_image2 = DYImage(f"{root_location}/ref/{init_file}.png").scaled( scale_image )
print("Quantized")
quantized_image = source_image.quantized(color_count, 1)
print("Cleanup, Quantize")
if cluster > 1 and use_map: quantized_image = quantized_image.cleanup_map(DYImage(f"{root_location}/ref/{init_file}.map.png"),3,cluster).rgb().quantized(color_count) 
if cluster > 1 and not use_map: quantized_image = quantized_image.cleanup(cluster).quantized(color_count,1)
print("Pixalated")
if pixel_step > 1: quantized_image = quantized_image.pixilated((pixel_step*k,pixel_step*k), 0, 1.0, False)
palette = quantized_image.palette()
masks = quantized_image.extract(palette)
rounded_masks = []
for mask in tqdm(masks): rounded_masks.append( mask.rounded(8, 110))
stencil_masks = []
stencil_masks.append( DYImage(f"{root_location}/tex/tex1.png").scaled( scale_image * msk_m).expanded(source_image.size()))
stencil_masks.append( DYImage(f"{root_location}/tex/tex2.png").scaled( scale_image * msk_m ).expanded(source_image.size()))
for i, color in enumerate(tqdm(palette)): source_image.paint(masks[i], color)
source_image.save(f"{root_location}/out/{init_file}.blobs.v{version}.png")
for i, color in enumerate(tqdm(palette)): source_image.paint(rounded_masks[i].blur(12*2).clamp(10).multed(stencil_masks[random.randint(0,number_of_masks-1)],threshold=False), color, amount=0.5, threshold=False, volume_diff=0.03)
source_image2.blend(source_image, 0.5)
source_image2 = source_image2.mult(1.05)
for i, color in enumerate(tqdm(palette)): source_image.comp(source_image2, rounded_masks[i].multed(stencil_masks[random.randint(0,number_of_masks-1)]))
source_image = source_image.glow_blur(0.0, 15, glow_amount)
source_image.save(f"{root_location}/out/{init_file}.painted.v{version}.png") 
source_image = source_image.sharpen()
source_image.save(f"{root_location}/out/{init_file}.sharpened.v{version}.png") 