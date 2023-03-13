import os
import re

import cv2

path = "/workspace/MVDeTr_research/output"
# path = '/workspace/MVDeTr_research/imgs'
paths = [os.path.join(path, f"{i:08}.png") for i in range(1, 41)]

fps = 5
frame_array = []

for file in paths:
    img = cv2.imread(file)
    height, width, layers = img.shape
    size = (width, height)
    print(file)
    frame_array.append(img)

out = cv2.VideoWriter(
    os.path.join(path, "mvdet_sort.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, size
)
# out = cv2.VideoWriter(os.path.join(path, 'camera_view.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

for i in range(len(frame_array)):
    out.write(frame_array[i])

out.release()
