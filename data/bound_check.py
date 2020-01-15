from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import Imath
import numpy as np
import OpenEXR
import os
import sys

input_names = np.array(["RockDetailMask.RockDetailMask", "SoftDetailMask.SoftDetailMask", "cliffs.cliffs", "height.height", "mesa.mesa", "slope.slope", "slopex.slopex", "slopez.slopez"])
output_names = np.array(["RockDetailMask.RockDetailMask", "SoftDetailMask.SoftDetailMask", "bedrock.bedrock", "cliffs.cliffs", "flow.flow", "flowx.flowx", "flowz.flowz", "height.height", "mesa.mesa", "sediment.sediment", "water.water"])
input_channels = np.array([3]) #height, mesa, slope
output_channels = np.array([7]) #flow, height

def read_exr_float32(filename, channel_names, height, width):
    exr_file = OpenEXR.InputFile(filename)
    data_list = exr_file.channels(channel_names)
    return np.array([np.frombuffer(data, np.float32)for data in data_list]).reshape(-1, height, width).transpose(1, 2, 0)

low = 99999
high = -99999
for file in os.listdir(sys.argv[1]):
    if file[-4:] != '.exr':
        continue
    exr_file = read_exr_float32(os.path.join(sys.argv[1], file), ["height.height"], 512, 512)
    low = min(low, np.min(exr_file))
    high = max(high, np.max(exr_file))

print(low)
print(high)