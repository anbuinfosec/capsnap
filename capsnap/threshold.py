from .image import Image

def threshold_global(image: Image, threshold: int, invert: bool = False) -> Image:
    """
    Applies a global threshold to a grayscale image.
    Pixels >= threshold become 255, else 0.
    If invert is True, pixels < threshold become 255 (foreground text is 255).
    """
    if image.channels != 1:
        raise ValueError("Thresholding requires a grayscale image")
        
    out = bytearray(len(image.pixels))
    pixels = image.pixels
    
    if invert:
        for i in range(len(pixels)):
            out[i] = 255 if pixels[i] < threshold else 0
    else:
        for i in range(len(pixels)):
            out[i] = 255 if pixels[i] >= threshold else 0
            
    return Image(image.width, image.height, 1, out)

def otsu_threshold_value(image: Image) -> int:
    """
    Calculates the optimal threshold value using Otsu's method.
    """
    if image.channels != 1:
        raise ValueError("Otsu's method requires a grayscale image")
        
    histogram = [0] * 256
    pixels = image.pixels
    
    for p in pixels:
        histogram[p] += 1
        
    total = len(pixels)
    
    sum_all = sum(i * histogram[i] for i in range(256))
    
    sumB = 0
    wB = 0
    wF = 0
    
    varMax = 0.0
    threshold = 0
    
    for t in range(256):
        wB += histogram[t]
        if wB == 0:
            continue
            
        wF = total - wB
        if wF == 0:
            break
            
        sumB += t * histogram[t]
        
        mB = sumB / wB
        mF = (sum_all - sumB) / wF
        
        varBetween = wB * wF * (mB - mF) ** 2
        
        if varBetween > varMax:
            varMax = varBetween
            threshold = t
            
    # The threshold t separates [0, t] and [t+1, 255]. 
    # For a > or >= comparison where we want [t+1, 255] to be foreground,
    # the threshold value should be t + 1.
    return threshold + 1

def threshold_otsu(image: Image, invert: bool = False) -> Image:
    """
    Applies Otsu's thresholding to a grayscale image.
    """
    t = otsu_threshold_value(image)
    return threshold_global(image, t, invert)

def threshold_adaptive(image: Image, block_size: int = 15, c: int = 10, invert: bool = False) -> Image:
    """
    Applies adaptive thresholding using the mean of a local neighborhood.
    block_size should be odd.
    c is a constant subtracted from the mean.
    """
    if image.channels != 1:
        raise ValueError("Adaptive thresholding requires a grayscale image")
    
    if block_size % 2 == 0:
        block_size += 1
        
    width = image.width
    height = image.height
    pixels = image.pixels
    out = bytearray(len(pixels))
    
    # Compute integral image for fast local mean calculation
    integral = [0] * (width * height)
    
    for y in range(height):
        sum_row = 0
        for x in range(width):
            idx = y * width + x
            sum_row += pixels[idx]
            if y == 0:
                integral[idx] = sum_row
            else:
                integral[idx] = integral[(y - 1) * width + x] + sum_row
                
    half = block_size // 2
    
    for y in range(height):
        for x in range(width):
            x1 = max(0, x - half)
            y1 = max(0, y - half)
            x2 = min(width - 1, x + half)
            y2 = min(height - 1, y + half)
            
            count = (x2 - x1 + 1) * (y2 - y1 + 1)
            
            # Sum over the region (x1, y1) to (x2, y2)
            sum_region = integral[y2 * width + x2]
            if x1 > 0:
                sum_region -= integral[y2 * width + (x1 - 1)]
            if y1 > 0:
                sum_region -= integral[(y1 - 1) * width + x2]
            if x1 > 0 and y1 > 0:
                sum_region += integral[(y1 - 1) * width + (x1 - 1)]
                
            mean = sum_region // count
            threshold = mean - c
            
            val = pixels[y * width + x]
            idx = y * width + x
            
            if invert:
                out[idx] = 255 if val < threshold else 0
            else:
                out[idx] = 255 if val >= threshold else 0
                
    return Image(width, height, 1, out)
