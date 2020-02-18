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

negative = 0
positive = 0
for file1, file2 in zip(os.listdir(sys.argv[1]), os.listdir(sys.argv[2])):
    if file1[-4:] != '.exr':
        continue
    exr_file_original = read_exr_float32(os.path.join(sys.argv[1], file1), ["1"], 512, 512)
    exr_file_generated = read_exr_float32(os.path.join(sys.argv[1], file2), ["1"], 512, 512)
    diff = (exr_file_generated - exr_file_original)
    negative += len(diff[diff<0])
    positive += len(diff[diff>0])

print(negative)
print(positive)