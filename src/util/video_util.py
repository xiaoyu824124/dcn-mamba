# Last modified: 2025-10-17

from typing import List, Union
import os  # 新增引用，用于检查文件路径
import matplotlib
import numpy as np
import torch
from moviepy.editor import ImageSequenceClip
from PIL import Image
import logging


def generate_gif_from_images(
    images_ls: List[Union[Image.Image, np.ndarray]],
    output_path: str,
    duration=500,
    loop=0,
    optimize=False,
):
    """
    Generate a GIF from multiple images.
    """
    pil_image_ls = []
    for img in images_ls:
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        pil_image_ls.append(img)

    # Save images as an animated GIF
    pil_image_ls[0].save(
        output_path,
        save_all=True,
        append_images=pil_image_ls[1:],
        format="GIF",
        duration=duration,
        loop=loop,
        optimize=optimize,
    )


def generate_video_from_image_paths(
    image_paths: List[str],
    output_path: str,
    fps: int = 30,
    bitrate: int = 1000,
    verbose=True,
):
    """
    Create a video from a list of image file paths using MoviePy.
    """
    if not image_paths:
        raise ValueError("The list of image paths is empty.")

    # --- [修复开始]：过滤非法路径和文件夹 ---
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')
    valid_paths = [
        p for p in image_paths 
        if os.path.isfile(p) and p.lower().endswith(valid_extensions)
    ]
    
    # 重新排序确保帧顺序正确
    valid_paths.sort()

    if not valid_paths:
        # 如果过滤完是空的，说明传入的全是文件夹或非图片，这能帮我们定位问题
        print(f"Warning: Original paths count: {len(image_paths)}, but no valid images found.")
        print(f"First few raw paths: {image_paths[:5]}")
        raise ValueError("No valid image files found in the provided list.")
        
    if verbose:
        print(f"Processing {len(valid_paths)} valid images for video generation.")
    # --- [修复结束] ---

    # Create a clip from the image sequence
    clip = ImageSequenceClip(valid_paths, fps=fps)

    # Set the duration explicitly
    clip = clip.set_duration(len(valid_paths) / fps)

    # --- [优化]：同时保证宽和高都是偶数，避免编码报错 ---
    new_h = clip.h - (clip.h % 2)
    new_w = clip.w - (clip.w % 2)
    if new_h != clip.h or new_w != clip.w:
        clip = clip.crop(x1=0, y1=0, x2=new_w, y2=new_h)

    # Write the clip to a video file
    clip.write_videofile(
        output_path,
        codec="libx264",
        bitrate=f"{bitrate}K",
        # ffmpeg_params=["-profile:v", "high", "-pix_fmt", "yuv420p"],
        verbose=verbose,
        logger=None if not verbose else "bar",
    )


def generate_video_from_images(images, output_path, fps=24, bitrate=1000, verbose=True):
    """
    Generate a video from multiple images using moviepy.
    """
    # Convert PIL images to NumPy arrays if necessary
    images = [np.array(img) if isinstance(img, Image.Image) else img for img in images]

    # Fix dimensions in-place
    for i, image in enumerate(images):
        height, width = image.shape[:2]

        # Handle odd height
        if height % 2 != 0:
            image = image[1:]

        # Handle odd width
        if width % 2 != 0:
            image = image[:, 1:]

        images[i] = image

    # Create a video clip from the image sequence
    clip = ImageSequenceClip(images, fps=fps)

    # Set the duration explicitly
    clip = clip.set_duration(len(images) / fps)

    try:
        # Write the video file with the specified bitrate
        clip.write_videofile(
            output_path,
            codec="libx264",
            bitrate=f"{bitrate}k",
            fps=fps,
            verbose=verbose,
            logger=None if not verbose else "bar",
            preset="medium",  # Balanced preset for compatibility
            audio=False,  # No audio needed for image sequence
            threads=4,  # Limit threads to prevent memory issues
            ffmpeg_params=[
                "-pix_fmt",
                "yuv420p",  # Ensure compatibility with QuickTime
                "-movflags",
                "+faststart",  # Enable streaming optimization
            ],
        )
    finally:
        clip.close()

    return output_path


def colorize_depth_maps(
    depth_map, min_depth, max_depth, cmap="plasma_r", valid_mask=None
):
    """
    Args:
        depth_map:
        cmap: matplotlib color map
    """
    assert len(depth_map.shape) >= 2, "Invalid dimension"

    if isinstance(depth_map, torch.Tensor):
        depth = depth_map.detach().clone().squeeze().numpy()
    elif isinstance(depth_map, np.ndarray):
        depth = depth_map.copy().squeeze()
    # reshape to [ (B,) H, W ]
    if depth.ndim < 3:
        depth = depth[np.newaxis, :, :]

    # colorize
    cm = matplotlib.colormaps[cmap]
    depth = ((depth - min_depth) / (max_depth - min_depth)).clip(0, 1)
    img_colored_np = cm(depth, bytes=False)[:, :, :, 0:3]  # value from 0 to 1
    img_colored_np = np.rollaxis(img_colored_np, 3, 1)

    if valid_mask is not None:
        if isinstance(depth_map, torch.Tensor):
            valid_mask = valid_mask.detach().numpy()
        valid_mask = valid_mask.squeeze()  # [H, W] or [B, H, W]
        if valid_mask.ndim < 3:
            valid_mask = valid_mask[np.newaxis, np.newaxis, :, :]
        else:
            valid_mask = valid_mask[:, np.newaxis, :, :]
        valid_mask = np.repeat(valid_mask, 3, axis=1)
        img_colored_np[~valid_mask] = 0

    if isinstance(depth_map, torch.Tensor):
        img_colored = torch.from_numpy(img_colored_np).float()
    elif isinstance(depth_map, np.ndarray):
        img_colored = img_colored_np

    return img_colored


def combine_images_side_by_side(image1, image2, middle_margin=0):
    # Get the sizes of the images
    width1, height1 = image1.size
    width2, height2 = image2.size

    # Create a new image with the combined width and the maximum height, including margin
    combined_width = width1 + width2 + middle_margin
    combined_height = max(height1, height2)

    combined_image = Image.new("RGB", (combined_width, combined_height))

    # Paste the images into the combined image
    combined_image.paste(image1, (0, 0))
    combined_image.paste(image2, (width1 + middle_margin, 0))

    return combined_image
