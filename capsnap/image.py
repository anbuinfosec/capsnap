class Image:
    """
    Internal representation of an image.
    Uses a 1D bytearray for pixel storage.
    Pixels are stored row by row, from top to bottom.
    """
    def __init__(self, width: int, height: int, channels: int, pixels: bytes | bytearray):
        if width <= 0 or height <= 0:
            raise ValueError("Image dimensions must be positive")
        if channels not in (1, 3, 4):
            raise ValueError("Channels must be 1 (grayscale), 3 (RGB), or 4 (RGBA)")
        
        expected_size = width * height * channels
        if len(pixels) != expected_size:
            raise ValueError(f"Expected {expected_size} bytes of pixel data, got {len(pixels)}")
            
        self.width = width
        self.height = height
        self.channels = channels
        self.pixels = bytearray(pixels)
        
    def get_pixel(self, x: int, y: int) -> tuple[int, ...]:
        """Gets the pixel value at (x, y) as a tuple of channel values."""
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise IndexError("Pixel coordinates out of bounds")
        idx = (y * self.width + x) * self.channels
        if self.channels == 1:
            return (self.pixels[idx],)
        elif self.channels == 3:
            return (self.pixels[idx], self.pixels[idx+1], self.pixels[idx+2])
        else: # channels == 4
            return (self.pixels[idx], self.pixels[idx+1], self.pixels[idx+2], self.pixels[idx+3])
            
    def set_pixel(self, x: int, y: int, value: tuple[int, ...]):
        """Sets the pixel value at (x, y)."""
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise IndexError("Pixel coordinates out of bounds")
        if len(value) != self.channels:
            raise ValueError(f"Expected {self.channels} channel values, got {len(value)}")
            
        idx = (y * self.width + x) * self.channels
        for c in range(self.channels):
            self.pixels[idx + c] = value[c]
            
    def clone(self) -> 'Image':
        """Returns a deep copy of the image."""
        return Image(self.width, self.height, self.channels, self.pixels.copy())
