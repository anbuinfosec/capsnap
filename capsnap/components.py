from .image import Image

class BoundingBox:
    def __init__(self, x: int, y: int, w: int, h: int):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        
    def area(self):
        return self.w * self.h

    @property
    def right(self):
        return self.x + self.w
        
    @property
    def bottom(self):
        return self.y + self.h
        
    def __repr__(self):
        return f"BoundingBox(x={self.x}, y={self.y}, w={self.w}, h={self.h})"

def find_components(image: Image) -> list[BoundingBox]:
    """
    Finds connected components in a binary image.
    Foreground pixels should be 255.
    Returns a list of BoundingBoxes.
    Uses an iterative DFS approach.
    """
    if image.channels != 1:
        raise ValueError("Connected components requires a binary image")
        
    width = image.width
    height = image.height
    pixels = image.pixels
    
    visited = bytearray(width * height)
    boxes = []
    
    # 8-connectivity
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    for y in range(height):
        for x in range(width):
            idx = y * width + x
            if pixels[idx] == 255 and visited[idx] == 0:
                # Start new component
                stack = [(x, y)]
                visited[idx] = 1
                
                min_x = x
                max_x = x
                min_y = y
                max_y = y
                
                while stack:
                    cx, cy = stack.pop()
                    
                    if cx < min_x: min_x = cx
                    if cx > max_x: max_x = cx
                    if cy < min_y: min_y = cy
                    if cy > max_y: max_y = cy
                    
                    for dx, dy in dirs:
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < width and 0 <= ny < height:
                            nidx = ny * width + nx
                            if pixels[nidx] == 255 and visited[nidx] == 0:
                                visited[nidx] = 1
                                stack.append((nx, ny))
                                
                boxes.append(BoundingBox(min_x, min_y, max_x - min_x + 1, max_y - min_y + 1))
                
    return boxes
