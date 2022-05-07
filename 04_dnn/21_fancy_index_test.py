# ----------------------------------------------------------
# Test fancy index function 
# 
#  color_mask = np.array(colors, dtype=np.uint8)[mask]
# ----------------------------------------------------------
import numpy as np

# segmentation mask (1=human, 2=car)
mask = np.array([[0, 0, 0, 0, 0],
                 [0, 1, 1, 0, 0],
                 [0, 0, 1, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 2, 2],])
print(f'mask={mask}')

# color table for masking overlay
colors = np.array([[0,      0,      0],
                   [255,    255,    0],
                   [0,      255,    255],
                   [255,    0,      255],])
print(f'colors={colors}')
print(f'colors[0]={colors[0]}')
print(f'colors[3]={colors[3]}')

# fancy index(convert to BGR image array format)
color_mask = np.array(colors, dtype=np.uint8)[mask]
print(f'color_mask={color_mask}')
print(f'color_mask[1]={color_mask[1]}')
