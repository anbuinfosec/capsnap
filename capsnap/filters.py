from .image import Image

def median_filter(image: Image, kernel_size: int = 3) -> Image:
    """
    Applies a median filter to a grayscale image, useful for removing salt-and-pepper noise.
    """
    if image.channels != 1:
        raise ValueError("Median filter requires a grayscale image")
        
    width = image.width
    height = image.height
    pixels = image.pixels
    out = bytearray(len(pixels))
    
    half = kernel_size // 2
    
    for y in range(height):
        for x in range(width):
            neighbors = []
            for ky in range(-half, half + 1):
                ny = y + ky
                if 0 <= ny < height:
                    for kx in range(-half, half + 1):
                        nx = x + kx
                        if 0 <= nx < width:
                            neighbors.append(pixels[ny * width + nx])
            neighbors.sort()
            out[y * width + x] = neighbors[len(neighbors) // 2]
            
    return Image(width, height, 1, out)
