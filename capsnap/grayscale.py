from .image import Image

def to_grayscale(image: Image) -> Image:
    """
    Converts an Image to 8-bit grayscale.
    Uses luminance formula: Y = 0.299R + 0.587G + 0.114B
    If the image is already grayscale, it returns a clone.
    """
    if image.channels == 1:
        return image.clone()
        
    out_pixels = bytearray(image.width * image.height)
    pixels = image.pixels
    
    if image.channels == 3:
        for i in range(0, len(pixels), 3):
            # Y = (R * 299 + G * 587 + B * 114) / 1000
            y = (pixels[i] * 299 + pixels[i+1] * 587 + pixels[i+2] * 114) // 1000
            out_pixels[i // 3] = y
    elif image.channels == 4:
        for i in range(0, len(pixels), 4):
            # Ignore alpha for now, or blend with white background?
            # Usually OCR assumes white background if transparent.
            alpha = pixels[i+3]
            r = pixels[i]
            g = pixels[i+1]
            b = pixels[i+2]
            
            # Blend with white: Color = (Color * Alpha + White * (255 - Alpha)) / 255
            r = (r * alpha + 255 * (255 - alpha)) // 255
            g = (g * alpha + 255 * (255 - alpha)) // 255
            b = (b * alpha + 255 * (255 - alpha)) // 255
            
            y = (r * 299 + g * 587 + b * 114) // 1000
            out_pixels[i // 4] = y
            
    return Image(image.width, image.height, 1, out_pixels)
