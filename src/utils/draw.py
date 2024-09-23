from __future__ import annotations
import numpy as np
import cv2
import torch

def draw_mask_and_save(img:np.ndarray, pred:torch.Tensor, save_path:str='./img/1/example.png')->None:
    color_list = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255)]
    if len(pred.shape) == 2:
        pred = pred.unsqueeze(0)
    if len(img.shape) == 2: # If img is HxW
        img = np.expand_dims(img, axis=-1)  # Convert to HxWx1
    if len(img) == 1: # If img is 1xHxW
        img = np.tile(img, (1, 1, 3))  # Convert to HxWx3 by repeating the channel

    h, w, _ = img.shape
    rgb = np.zeros((h, w, 3))
    mask = np.ones((h, w, 3))
    classes = len(pred)
    pred = pred.cpu().numpy()
    for i in reversed(range(classes)):
        mask[pred[i] == 1] = (0.5, 0.5, 0.5)
        rgb[pred[i] == 1] = color_list[i]
    res = (img+rgb)*mask
    res = res.astype(np.uint8)
    res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, res)