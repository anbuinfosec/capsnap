from .image import Image

def erode(image: Image, kernel_size: int = 3) -> Image:
    """
    Applies erosion to a binary image.
    Pixels become 255 only if all pixels under the kernel are 255.
    Otherwise they become 0.
    """
    if image.channels != 1:
        raise ValueError("Morphology requires a grayscale/binary image")
        
    width = image.width
    height = image.height
    pixels = image.pixels
    out = bytearray(len(pixels))
    
    half = kernel_size // 2
    
    for y in range(height):
        for x in range(width):
            min_val = 255
            for ky in range(-half, half + 1):
                ny = y + ky
                if 0 <= ny < height:
                    for kx in range(-half, half + 1):
                        nx = x + kx
                        if 0 <= nx < width:
                            val = pixels[ny * width + nx]
                            if val < min_val:
                                min_val = val
            out[y * width + x] = min_val
            
    return Image(width, height, 1, out)

def dilate(image: Image, kernel_size: int = 3) -> Image:
    """
    Applies dilation to a binary image.
    Pixels become 255 if any pixel under the kernel is 255.
    Otherwise they become 0.
    """
    if image.channels != 1:
        raise ValueError("Morphology requires a grayscale/binary image")
        
    width = image.width
    height = image.height
    pixels = image.pixels
    out = bytearray(len(pixels))
    
    half = kernel_size // 2
    
    for y in range(height):
        for x in range(width):
            max_val = 0
            for ky in range(-half, half + 1):
                ny = y + ky
                if 0 <= ny < height:
                    for kx in range(-half, half + 1):
                        nx = x + kx
                        if 0 <= nx < width:
                            val = pixels[ny * width + nx]
                            if val > max_val:
                                max_val = val
            out[y * width + x] = max_val
            
    return Image(width, height, 1, out)

def opening(image: Image, kernel_size: int = 3) -> Image:
    """Erosion followed by dilation. Useful for removing small noise."""
    eroded = erode(image, kernel_size)
    return dilate(eroded, kernel_size)

def closing(image: Image, kernel_size: int = 3) -> Image:
    """Dilation followed by erosion. Useful for closing small holes."""
    dilated = dilate(image, kernel_size)
    return erode(dilated, kernel_size)
