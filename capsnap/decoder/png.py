import struct
import zlib
from ..image import Image

PNG_SIGNATURE = b'\x89PNG\r\n\x1a\n'

def paeth_predictor(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    elif pb <= pc:
        return b
    else:
        return c

class PNGError(Exception):
    pass

def read_png(data: bytes) -> Image:
    if not data.startswith(PNG_SIGNATURE):
        raise PNGError("Invalid PNG signature")
        
    offset = 8
    
    width = 0
    height = 0
    bit_depth = 0
    color_type = 0
    compression = 0
    filter_method = 0
    interlace = 0
    
    idat_data = bytearray()
    
    while offset < len(data):
        if offset + 8 > len(data):
            break
        length = struct.unpack(">I", data[offset:offset+4])[0]
        chunk_type = data[offset+4:offset+8]
        offset += 8
        
        chunk_data = data[offset:offset+length]
        offset += length
        
        # skip CRC
        offset += 4
        
        if chunk_type == b'IHDR':
            (width, height, bit_depth, color_type, compression, 
             filter_method, interlace) = struct.unpack(">IIBBBBB", chunk_data)
            
            if bit_depth not in (1, 8):
                raise PNGError(f"Unsupported bit depth: {bit_depth}. Only 1-bit and 8-bit are supported.")
            if interlace != 0:
                raise PNGError("Interlaced PNGs are not supported.")
            if color_type not in (0, 2, 6): # grayscale, truecolor, truecolor+alpha
                raise PNGError(f"Unsupported color type: {color_type}. Expected 0 (gray), 2 (RGB), or 6 (RGBA).")
            if bit_depth == 1 and color_type != 0:
                raise PNGError("1-bit depth is only supported for grayscale (color type 0).")
                
        elif chunk_type == b'IDAT':
            idat_data.extend(chunk_data)
        elif chunk_type == b'IEND':
            break
            
    if not idat_data:
        raise PNGError("No IDAT chunks found")
        
    try:
        decompressed = zlib.decompress(idat_data)
    except zlib.error as e:
        raise PNGError(f"zlib decompression failed: {e}")
        
    # Determine channels and stride
    if color_type == 0:
        channels = 1
    elif color_type == 2:
        channels = 3
    elif color_type == 6:
        channels = 4
    else:
        channels = 1
        
    if bit_depth == 1:
        stride = (width + 7) // 8
        bpp = 1
    else:
        stride = width * channels
        bpp = channels
    
    if len(decompressed) != height * (stride + 1):
        raise PNGError(f"Decompressed data size {len(decompressed)} does not match expected {height * (stride + 1)}")
        
    unfiltered = bytearray(height * stride)
    
    # Unfilter
    for y in range(height):
        filter_type = decompressed[y * (stride + 1)]
        scanline_start = y * (stride + 1) + 1
        scanline = decompressed[scanline_start:scanline_start + stride]
        
        out_start = y * stride
        
        if filter_type == 0: # None
            unfiltered[out_start:out_start+stride] = scanline
        elif filter_type == 1: # Sub
            for i in range(stride):
                a = unfiltered[out_start + i - bpp] if i >= bpp else 0
                unfiltered[out_start + i] = (scanline[i] + a) & 0xff
        elif filter_type == 2: # Up
            for i in range(stride):
                b = unfiltered[out_start - stride + i] if y > 0 else 0
                unfiltered[out_start + i] = (scanline[i] + b) & 0xff
        elif filter_type == 3: # Average
            for i in range(stride):
                a = unfiltered[out_start + i - bpp] if i >= bpp else 0
                b = unfiltered[out_start - stride + i] if y > 0 else 0
                unfiltered[out_start + i] = (scanline[i] + (a + b) // 2) & 0xff
        elif filter_type == 4: # Paeth
            for i in range(stride):
                a = unfiltered[out_start + i - bpp] if i >= bpp else 0
                b = unfiltered[out_start - stride + i] if y > 0 else 0
                c = unfiltered[out_start - stride + i - bpp] if y > 0 and i >= bpp else 0
                unfiltered[out_start + i] = (scanline[i] + paeth_predictor(a, b, c)) & 0xff
        else:
            raise PNGError(f"Unknown filter type: {filter_type}")
            
    # Convert to 8-bit pixels
    pixels = bytearray(width * height * channels)
    if bit_depth == 1:
        for y in range(height):
            for x in range(width):
                byte_idx = y * stride + (x // 8)
                bit_idx = 7 - (x % 8)
                bit = (unfiltered[byte_idx] >> bit_idx) & 1
                # 0 -> 0 (black), 1 -> 255 (white)
                pixels[y * width + x] = 255 if bit else 0
    else:
        pixels[:] = unfiltered
        
    return Image(width, height, channels, pixels)

def write_png(image: Image) -> bytes:
    """Writes an Image to PNG format (unfiltered)."""
    out = bytearray(PNG_SIGNATURE)
    
    def write_chunk(chunk_type: bytes, data: bytes):
        out.extend(struct.pack(">I", len(data)))
        out.extend(chunk_type)
        out.extend(data)
        crc = zlib.crc32(chunk_type + data) & 0xffffffff
        out.extend(struct.pack(">I", crc))
        
    if image.channels == 1:
        color_type = 0
    elif image.channels == 3:
        color_type = 2
    elif image.channels == 4:
        color_type = 6
    else:
        raise ValueError("Unsupported channel count")
        
    # IHDR
    ihdr_data = struct.pack(">IIBBBBB", image.width, image.height, 8, color_type, 0, 0, 0)
    write_chunk(b'IHDR', ihdr_data)
    
    # Format scanlines with filter byte 0 (None)
    stride = image.width * image.channels
    scanlines = bytearray()
    for y in range(image.height):
        scanlines.append(0) # Filter byte 0
        start = y * stride
        scanlines.extend(image.pixels[start:start+stride])
        
    # IDAT
    compressed = zlib.compress(scanlines)
    write_chunk(b'IDAT', compressed)
    
    # IEND
    write_chunk(b'IEND', b'')
    
    return bytes(out)
