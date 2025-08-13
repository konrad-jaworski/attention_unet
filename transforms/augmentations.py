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
                    label = torch.flip(label, dims=[axis])
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

        static_mask = False
        if mask is not None:
            if mask.ndim == 3:
                static_mask = True  # mask is [C, H, W], not a sequence
            elif mask.ndim != 4:
                raise ValueError(f"Mask must be 3D or 4D, got {mask.shape}")

        # Pick one random angle and shift for all frames
        angle = random.uniform(*self.degrees)
        dx = random.uniform(-self.max_shift[0], self.max_shift[0])
        dy = random.uniform(-self.max_shift[1], self.max_shift[1])

        # Rotate video frames
        rotated_frames = [
            self.rotate_frame(video[:, t, :, :], angle, dx, dy, mode="bilinear")
            .unsqueeze(1)
            for t in range(video.shape[1])
        ]
        rotated_video = torch.cat(rotated_frames, dim=1)

        # Rotate mask
        rotated_mask = None
        if mask is not None:
            if static_mask:
                # Keep uint8 type for masks
                rotated_mask = self.rotate_frame(
                    mask.float(), angle, dx, dy, mode="nearest"
                ).to(mask.dtype)
            else:
                rotated_mask_frames = [
                    self.rotate_frame(mask[:, t, :, :].float(), angle, dx, dy, mode="nearest")
                    .unsqueeze(1)
                    for t in range(mask.shape[1])
                ]
                rotated_mask = torch.cat(rotated_mask_frames, dim=1).to(mask.dtype)

        return rotated_video, rotated_mask

    def rotate_frame(self, frame, angle_deg, dx=0.0, dy=0.0, mode="bilinear"):
        C, H, W = frame.shape
        theta = math.radians(angle_deg)
        cos, sin = math.cos(theta), math.sin(theta)

        # Normalize shifts to [-1, 1]
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

        # Generate displacement field
        if self.same_for_sequence:
            displacement_field = self.generate_displacement(H, W, device=video.device, dtype=video.dtype)

        deformed_frames = []

        for t in range(T):
            if not self.same_for_sequence:
                displacement_field = self.generate_displacement(H, W, device=video.device, dtype=video.dtype)

            deformed = self.apply_displacement(video[:, t, :, :], displacement_field, mode='bilinear')
            deformed_frames.append(deformed.unsqueeze(1))

        deformed_video = torch.cat(deformed_frames, dim=1)
        return deformed_video, mask

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

class PrependFirstFrame:
    def __init__(self, max_percent=0.2):
        """
        max_percent: float, maximum percent of sequence length to prepend
        """
        if not (0 <= max_percent <= 1):
            raise ValueError("max_percent must be between 0 and 1")
        self.max_percent = max_percent

    def __call__(self, video, mask=None):
        """
        video: [C, T, H, W] tensor
        mask: optional [Cmask, T, H, W] or [T, H, W]
        Returns: (new_video, new_mask)
        """
        C, T, H, W = video.shape

        # Determine number of frames to prepend
        max_prepend = int(T * self.max_percent)
        if max_prepend == 0:
            return video, mask
        num_prepend = random.randint(1, max_prepend)

        # First frame copy
        first_frame = video[:, 0, :, :].unsqueeze(1)
        prepend_frames = first_frame.repeat(1, num_prepend, 1, 1)
        new_video = torch.cat([prepend_frames, video], dim=1)
        
        return new_video, mask

class RandomCropSequence:
    def __init__(self, crop_size=(256, 256)):
        """
        crop_size: (height, width) of the output crop
        """
        self.crop_h, self.crop_w = crop_size

    def __call__(self, video, mask=None):
        """
        video: [C, T, H, W]
        mask: optional [Cmask, T, H, W] or [T, H, W]
        """
        C, T, H, W = video.shape
        if self.crop_h > H or self.crop_w > W:
            raise ValueError(f"Crop size {self.crop_h}x{self.crop_w} is larger than video size {H}x{W}")

        # Random crop coordinates
        top = random.randint(0, H - self.crop_h)
        left = random.randint(0, W - self.crop_w)

        # Apply crop to all frames
        video_cropped = video[:, :, top:top + self.crop_h, left:left + self.crop_w]

        # Crop mask if provided
        if mask is not None:
            if mask.ndim == 3:  # [T, H, W]
                mask_cropped = mask[:, top:top + self.crop_h, left:left + self.crop_w]
            elif mask.ndim == 4:  # [Cmask, T, H, W]
                mask_cropped = mask[:, :, top:top + self.crop_h, left:left + self.crop_w]
            else:
                raise ValueError("Mask must have shape [T,H,W] or [Cmask,T,H,W]")
        else:
            mask_cropped = None

        return video_cropped, mask_cropped


class RandomBrightnessContrast:
    def __init__(self, brightness_range=(-0.2, 0.2), contrast_range=(0.8, 1.2), p=0.5):
        """
        brightness_range: tuple, additive shift range (e.g., -0.2 to 0.2 for normalized data)
        contrast_range: tuple, multiplicative factor range (e.g., 0.8 to 1.2)
        p: probability of applying the transform
        """
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.p = p

    def __call__(self, video, mask=None):
        """
        video: Tensor [C, T, H, W]
        mask: Optional mask tensor [T, H, W] or [1, T, H, W] (unchanged)
        """
        if random.random() > self.p:
            return video, mask

        # Sample brightness and contrast factors
        brightness_shift = random.uniform(*self.brightness_range)
        contrast_factor = random.uniform(*self.contrast_range)

        # Apply per-sequence (same for all frames)
        video_aug = video * contrast_factor + brightness_shift

        # If data is in [0, 1], clip to valid range
        if torch.is_floating_point(video_aug):
            min_val, max_val = video.min().item(), video.max().item()
            if min_val >= 0 and max_val <= 1:
                video_aug = torch.clamp(video_aug, 0.0, 1.0)

        return video_aug, mask

class RandomPhaseAwareSpeedChange:
    def __init__(self, speed_range=(0.5, 1.5), p=0.5):
        """
        speed_range: (min_factor, max_factor) for each phase
        p: probability of applying the transform
        """
        self.speed_range = speed_range
        self.p = p

    def __call__(self, video, mask=None):
        """
        video: Tensor [C, T, H, W]
        mask: Optional static mask [H, W] or [1, H, W]
        """
        if random.random() > self.p:
            return video, mask

        C, T, H, W = video.shape
        device = video.device

        # 1. Find the frame with highest average value
        avg_values = video.mean(dim=(0, 2, 3))  # mean over C,H,W → [T]
        peak_idx = torch.argmax(avg_values).item()

        # 2. Random speed factors for heating and cooling
        heating_factor = random.uniform(*self.speed_range)
        cooling_factor = random.uniform(*self.speed_range)

        # 3. Interpolate heating phase
        heating_len = peak_idx + 1
        heating_time_idx = torch.linspace(0, heating_len - 1, 
                                          int(heating_len / heating_factor),
                                          device=device)

        heating_part = F.interpolate(
            video[:, :heating_len, :, :].unsqueeze(0),
            size=(len(heating_time_idx), H, W),
            mode='trilinear',
            align_corners=False
        ).squeeze(0)

        # 4. Interpolate cooling phase
        cooling_len = T - peak_idx - 1
        if cooling_len > 0:
            cooling_time_idx = torch.linspace(0, cooling_len - 1, 
                                              int(cooling_len / cooling_factor),
                                              device=device)
            cooling_part = F.interpolate(
                video[:, peak_idx+1:, :, :].unsqueeze(0),
                size=(len(cooling_time_idx), H, W),
                mode='trilinear',
                align_corners=False
            ).squeeze(0)
        else:
            cooling_part = torch.empty((C, 0, H, W), device=device)

        # 5. Combine parts and resample back to T frames
        combined = torch.cat([heating_part, cooling_part], dim=1)
        video_final = F.interpolate(
            combined.unsqueeze(0),
            size=(T, H, W),
            mode='trilinear',
            align_corners=False
        ).squeeze(0)

        # 6. Return video + original mask
        return video_final, mask

class NormalizeTo01:
    def __call__(self,image,label=None):
        min_val=image.min()
        max_val=image.max()
        image=(image-min_val)/(max_val-min_val+1e-5)
        return image, label