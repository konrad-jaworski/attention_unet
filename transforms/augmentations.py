import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
import torch.nn.functional as F
import random
import os
import json
import torchvision.transforms.functional as TF
import math

class Compose3D:
    """
    Class handling number of transformation applied to our data
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image,label=None):
        for t in self.transforms:
            image,label = t(image,label)
        return image, label

class RandomFlip3D:
    def __init__(self, axes=(0, 1, 2), p=0.5):
        self.axes = axes  # which axes to possibly flip: 0=H, 1=W, 2=D
        self.p = p

    def __call__(self, image, label=None):
        for axis in self.axes:
            if random.random() < self.p:
                image = torch.flip(image, dims=[axis + 1])  # +1 to skip channel dim
                if label is not None:
                    label = torch.flip(label, dims=[axis + 1])
        return image, label
    
class AddGaussianNoise3D:
    def __init__(self, mean=0.0, std=0.05):
        self.mean = mean
        self.std = std

    def __call__(self, image,label=None):
        noise = torch.randn_like(image) * self.std + self.mean
        aug_img=image + noise
        return aug_img,label

class RandomSequenceRotation:
    def __init__(self, degrees, max_shift=(0, 0), padding_mode='border'):
        """
        degrees: float or tuple
        max_shift: tuple (max_dx, max_dy) — maximum pixel shift in x and y directions
        padding_mode: 'zeros', 'border', or 'reflection'
        """
        if isinstance(degrees, (int, float)):
            self.degrees = (-degrees, degrees)
        else:
            self.degrees = degrees

        self.max_shift = max_shift

        if padding_mode not in ['zeros', 'border', 'reflection']:
            raise ValueError("padding_mode must be 'zeros', 'border', or 'reflection'")
        self.padding_mode = padding_mode

    def __call__(self, video, mask=None):
        if video.ndim != 4:
            raise ValueError(f"Video must be 4D [C, T, H, W], got {video.shape}")

        if mask is not None:
            if mask.ndim == 3:
                mask = mask.unsqueeze(0)
            if mask.ndim != 4:
                raise ValueError(f"Mask must be 3D or 4D, got {mask.shape}")

        # Pick one random angle for all frames
        angle = random.uniform(*self.degrees)
        # Pick one random shift for all frames
        dx = random.uniform(-self.max_shift[0], self.max_shift[0])
        dy = random.uniform(-self.max_shift[1], self.max_shift[1])

        rotated_frames = []
        rotated_mask_frames = [] if mask is not None else None

        for t in range(video.shape[1]):
            rotated = self.rotate_frame(video[:, t, :, :], angle, dx, dy, mode="bilinear")
            rotated_frames.append(rotated.unsqueeze(1))

            if mask is not None:
                rotated_m = self.rotate_frame(mask[:, t, :, :], angle, dx, dy, mode="nearest")
                rotated_mask_frames.append(rotated_m.unsqueeze(1))

        rotated_video = torch.cat(rotated_frames, dim=1)
        rotated_mask = torch.cat(rotated_mask_frames, dim=1) if mask is not None else None
        return rotated_video, rotated_mask

    def rotate_frame(self, frame, angle_deg, dx=0.0, dy=0.0, mode="bilinear"):
        C, H, W = frame.shape
        theta = math.radians(angle_deg)
        cos, sin = math.cos(theta), math.sin(theta)

        # Normalize shifts to [-1, 1] coordinates
        dx_norm = 2.0 * dx / W
        dy_norm = 2.0 * dy / H

        # Rotation + shift affine matrix
        rotation_matrix = torch.tensor([
            [cos, -sin, dx_norm],
            [sin,  cos, dy_norm]
        ], dtype=frame.dtype, device=frame.device)

        grid = F.affine_grid(rotation_matrix.unsqueeze(0), size=(1, C, H, W), align_corners=True)

        rotated = F.grid_sample(
            frame.unsqueeze(0),
            grid,
            mode=mode,
            padding_mode=self.padding_mode,
            align_corners=True
        )
        return rotated.squeeze(0)



class RandomElasticTransform:
    def __init__(self, alpha=5.0, sigma=3.0, padding_mode='border', same_for_sequence=True):
        """
        alpha: float, scaling factor for the displacement
        sigma: float, standard deviation for Gaussian smoothing of displacement
        padding_mode: 'zeros', 'border', 'reflection'
        same_for_sequence: bool, if True, same displacement applied to all frames
        """
        self.alpha = alpha
        self.sigma = sigma
        self.padding_mode = padding_mode
        self.same_for_sequence = same_for_sequence

    def __call__(self, video, mask=None):
        """
        video: [C, T, H, W] tensor
        mask: optional mask [Cmask, T, H, W] or [T, H, W]
        Returns: (deformed_video, deformed_mask)
        """
        C, T, H, W = video.shape

        if mask is not None:
            if mask.ndim == 3:
                mask = mask.unsqueeze(0)
            if mask.ndim != 4:
                raise ValueError(f"Mask must be 3D or 4D, got {mask.shape}")

        # Generate displacement field
        if self.same_for_sequence:
            displacement_field = self.generate_displacement(H, W, device=video.device, dtype=video.dtype)

        deformed_frames = []
        deformed_mask_frames = [] if mask is not None else None

        for t in range(T):
            if not self.same_for_sequence:
                displacement_field = self.generate_displacement(H, W, device=video.device, dtype=video.dtype)

            deformed = self.apply_displacement(video[:, t, :, :], displacement_field, mode='bilinear')
            deformed_frames.append(deformed.unsqueeze(1))

            if mask is not None:
                deformed_m = self.apply_displacement(mask[:, t, :, :], displacement_field, mode='nearest')
                deformed_mask_frames.append(deformed_m.unsqueeze(1))

        deformed_video = torch.cat(deformed_frames, dim=1)
        deformed_mask = torch.cat(deformed_mask_frames, dim=1) if mask is not None else None
        return deformed_video, deformed_mask

    def generate_displacement(self, H, W, device='cpu', dtype=torch.float32):
        # Random displacement in pixels
        dx = torch.randn(H, W, device=device, dtype=dtype)
        dy = torch.randn(H, W, device=device, dtype=dtype)

        # Smooth displacement with Gaussian convolution
        dx = self.smooth_displacement(dx, self.sigma)
        dy = self.smooth_displacement(dy, self.sigma)

        # Scale by alpha
        dx = dx * self.alpha / W
        dy = dy * self.alpha / H

        displacement = torch.stack((dx, dy), dim=-1)  # [H, W, 2]
        return displacement

    def smooth_displacement(self, disp, sigma):
        kernel_size = int(2 * sigma + 1)
        ax = torch.arange(kernel_size, device=disp.device, dtype=disp.dtype) - kernel_size // 2
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel /= kernel.sum()
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

        disp = disp.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        disp_smooth = F.conv2d(disp, kernel, padding=kernel_size//2)
        return disp_smooth.squeeze()

    def apply_displacement(self, frame, displacement, mode='bilinear'):
        """
        frame: [C, H, W]
        displacement: [H, W, 2] in normalized coordinates [-1,1]
        """
        C, H, W = frame.shape

        # Create base normalized grid
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=frame.device, dtype=frame.dtype),
            torch.linspace(-1, 1, W, device=frame.device, dtype=frame.dtype),
            indexing='ij'
        )
        base_grid = torch.stack((x, y), dim=-1)  # [H, W, 2]

        # Add displacement
        grid = base_grid + displacement

        grid = grid.unsqueeze(0)        # [1, H, W, 2]
        frame = frame.unsqueeze(0)      # [1, C, H, W]

        deformed = F.grid_sample(
            frame,
            grid,
            mode=mode,
            padding_mode=self.padding_mode,
            align_corners=True
        )
        return deformed.squeeze(0)


class NormalizeTo01:
    def __call__(self,image,label=None):
        min_val=image.min()
        max_val=image.max()
        image=(image-min_val)/(max_val-min_val+1e-5)
        return image, label