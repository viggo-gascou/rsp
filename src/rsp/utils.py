"""Utility functions used throughout the project."""

from typing import List, Optional, Tuple

import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw


def img_path_to_tensor(size: int, path: str, device: torch.device) -> torch.Tensor:
    """Convert a path to an image to a tensor.

    Args:
        size: Size of the image
        path: Path to the image
        device: Device to move the tensor to

    Returns:
        The tensor representation of the image.
    """
    img = Image.open(path).resize((size, size))
    return pil_to_tensor(img, device)


def tensor_to_pil(tensor_imgs: torch.Tensor) -> List[Image.Image]:
    """Convert a torch.Tensor to a list of PIL.Image.

    Args:
        tensor_imgs: Input tensor(s)

    Returns:
        A list of PIL.Images
    """
    tensor_imgs = (tensor_imgs / 2 + 0.5).clamp(0, 1)
    to_pil = T.ToPILImage()
    pil_imgs: List[Image.Image] = [to_pil(img) for img in tensor_imgs]
    return pil_imgs


def pil_to_tensor(
    pil_imgs: Image.Image | List[Image.Image], device: torch.device
) -> torch.Tensor:
    """Convert a PIL.Image or a list of PIL.Image to torch.Tensor.

    Args:
        pil_imgs: Input image(s)
        device: Device to move the tensor to

    Returns:
        A tensor of shape (N, C, H, W) where N is the number of images,
        C is the number of channels, H is the height, and W is the width.
    """
    to_torch = T.ToTensor()
    if isinstance(pil_imgs, Image.Image):
        tensor_imgs = to_torch(pil_imgs).unsqueeze(0) * 2 - 1
    elif isinstance(pil_imgs, list):
        tensor_imgs = torch.cat(
            [to_torch(pil_imgs).unsqueeze(0) * 2 - 1 for img in pil_imgs]
        ).to(device)

    return tensor_imgs


def add_margin(
    pil_img: Image.Image,
    top: int = 2,
    right: int = 2,
    bottom: int = 2,
    left: int = 2,
    color: Tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Add a margin to a PIL.Image.

    Args:
        pil_img: Input image
        top: Top margin
        right: Right margin
        bottom: Bottom margin
        left: Left margin
        color: Margin color

    Returns:
        A PIL.Image with margin
    """
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))

    return result


def image_grid(
    imgs: List[Image.Image] | Image.Image | torch.Tensor,
    rows: int = 1,
    cols: Optional[int] = None,
    size: Optional[int] = None,
    titles: Optional[str | List[str]] = None,
    top: int = 20,
    text_pos: Tuple[int, int] = (0, 0),
    add_margin_size: Optional[int] = None,
) -> Image.Image:
    """Create a grid of images.

    Args:
        imgs: List of images.
        rows: Number of rows in the grid.
        cols: Number of columns in the grid.
        size: Size of each image in the grid.
        titles: List of titles for each image.
        top: Top margin.
        font_size: Font size for the titles.
        text_pos: Position of the text.
        add_margin_size: Size of the margin to add to each image.

    Returns:
        A PIL.Image with the grid of images.
    """
    if isinstance(imgs, Image.Image):
        pil_images = [imgs]
    elif isinstance(imgs, torch.Tensor):
        pil_images = tensor_to_pil(imgs)
    elif isinstance(imgs, list) and all(isinstance(img, Image.Image) for img in imgs):
        pil_images = imgs
    else:
        raise TypeError(f"Invalid input type {type(imgs)}")

    if titles is not None:
        if isinstance(titles, str):
            titles = [titles]
        assert len(titles) == len(pil_images), (
            "Number of titles must match number of images"
        )

    if size is not None:
        pil_images = [img.resize((size, size)) for img in pil_images]
    if cols is None:
        cols = len(pil_images)
    assert len(pil_images) >= rows * cols
    if add_margin_size is not None:
        pil_images = [
            add_margin(
                img,
                top=add_margin_size,
                right=add_margin_size,
                bottom=add_margin_size,
                left=add_margin_size,
            )
            for img in pil_images
        ]

    w, h = pil_images[0].size
    delta = 0
    if len(pil_images) > 1 and not pil_images[1].size[1] == h:
        delta = h - pil_images[1].size[1]  # top
        h = pil_images[1].size[1]
    if titles is not None:
        h = top + h
    grid = Image.new("RGB", size=(cols * w, rows * h + delta))
    for i, img in enumerate(pil_images):
        if titles is not None:
            img = add_margin(img, top=top, bottom=0, left=0)
            draw = ImageDraw.Draw(img)
            draw.text(text_pos, titles[i], (0, 0, 0))
        if not delta == 0 and i > 0:
            grid.paste(img, box=(i % cols * w, i // cols * h + delta))
        else:
            grid.paste(img, box=(i % cols * w, i // cols * h))

    return grid
