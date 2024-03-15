from typing import Any, Mapping
import torch as pt
import numpy as np

def get_t_repetition(cfg: Mapping[str, Any]) -> pt.Tensor:
    t_repetition = (cfg.T - 2*(cfg.len_window - 1) - 1)//(cfg.stride+1)
    return t_repetition

def get_x_split(cfg: Mapping[str, Any], x: pt.Tensor) -> pt.Tensor:
    x_split = pt.stack([x[:, :, t*cfg.stride:t*cfg.stride+cfg.len_window] for t in range(cfg.t_repetition)], 1)
    return x_split.float()

class TubeMaskingGenerator:
    def __init__(self, input_size, mask_ratio):
        self.frames, self.num_patches_per_frame = input_size
        self.total_patches = self.frames * self.num_patches_per_frame 
        self.num_masks_per_frame = int(mask_ratio * self.num_patches_per_frame)
        self.total_masks = self.frames * self.num_masks_per_frame

    def __repr__(self):
        repr_str = "Masks: total patches {}, mask patches {}".format(
            self.total_patches, self.total_masks
        )
        return repr_str

    def __call__(self):
        mask_per_frame = np.hstack([
            np.zeros(self.num_patches_per_frame - self.num_masks_per_frame),
            np.ones(self.num_masks_per_frame),
        ])
        mask = np.tile(mask_per_frame, (self.frames, 1)).flatten()
        return mask 