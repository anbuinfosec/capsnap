from .image import Image
from .components import BoundingBox

def extract_patch(image: Image, box: BoundingBox) -> Image:
    """Extracts a sub-image defined by a BoundingBox."""
    if box.x < 0 or box.y < 0 or box.right > image.width or box.bottom > image.height:
        raise ValueError("Box is out of image bounds")
        
    out = bytearray(box.w * box.h * image.channels)
    pixels = image.pixels
    
    for y in range(box.h):
        src_y = box.y + y
        src_idx_start = (src_y * image.width + box.x) * image.channels
        src_idx_end = src_idx_start + box.w * image.channels
        
        dst_idx_start = y * box.w * image.channels
        out[dst_idx_start:dst_idx_start + box.w * image.channels] = pixels[src_idx_start:src_idx_end]
        
    return Image(box.w, box.h, image.channels, out)

def resize_nearest(image: Image, new_w: int, new_h: int) -> Image:
    """Resizes an image using nearest-neighbor interpolation."""
    out = bytearray(new_w * new_h * image.channels)
    pixels = image.pixels
    
    x_ratio = image.width / new_w
    y_ratio = image.height / new_h
    
    for y in range(new_h):
        src_y = int(y * y_ratio)
        for x in range(new_w):
            src_x = int(x * x_ratio)
            
            src_idx = (src_y * image.width + src_x) * image.channels
            dst_idx = (y * new_w + x) * image.channels
            
            for c in range(image.channels):
                out[dst_idx + c] = pixels[src_idx + c]
                
    return Image(new_w, new_h, image.channels, out)

def normalize_character(image: Image, target_size: int = 28) -> Image:
    """
    Normalizes a character image to a fixed square size while preserving aspect ratio.
    Pads the shorter dimension with 0 (assuming black background, white text).
    """
    # If the image is a binary image where text is 255 and bg is 0:
    w = image.width
    h = image.height
    
    # We want to scale the max dimension to target_size
    scale = target_size / max(w, h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    
    resized = resize_nearest(image, new_w, new_h)
    
    # Pad to target_size x target_size
    out = bytearray(target_size * target_size * image.channels)
    
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2
    
    for y in range(new_h):
        src_idx_start = (y * new_w) * image.channels
        src_idx_end = src_idx_start + new_w * image.channels
        
        dst_y = y + y_offset
        dst_idx_start = (dst_y * target_size + x_offset) * image.channels
        
        out[dst_idx_start:dst_idx_start + new_w * image.channels] = resized.pixels[src_idx_start:src_idx_end]
        
    return Image(target_size, target_size, image.channels, out)
