from .components import BoundingBox

class TextLine:
    def __init__(self, boxes: list[BoundingBox]):
        self.boxes = sorted(boxes, key=lambda b: b.x)
        if self.boxes:
            self.x = min(b.x for b in self.boxes)
            self.y = min(b.y for b in self.boxes)
            self.w = max(b.right for b in self.boxes) - self.x
            self.h = max(b.bottom for b in self.boxes) - self.y
        else:
            self.x = self.y = self.w = self.h = 0
            
    def __repr__(self):
        return f"TextLine(boxes={len(self.boxes)}, x={self.x}, y={self.y})"

def group_into_lines(boxes: list[BoundingBox]) -> list[TextLine]:
    """
    Groups character bounding boxes into text lines based on vertical overlap.
    """
    if not boxes:
        return []
        
    # Sort by y-coordinate
    sorted_boxes = sorted(boxes, key=lambda b: b.y)
    
    lines = []
    current_line_boxes = [sorted_boxes[0]]
    current_min_y = sorted_boxes[0].y
    current_max_y = sorted_boxes[0].bottom
    
    for box in sorted_boxes[1:]:
        # Check vertical overlap
        overlap = max(0, min(current_max_y, box.bottom) - max(current_min_y, box.y))
        box_height = box.h
        
        # If overlap is significant (e.g. > 30% of box height)
        if overlap > 0.3 * box_height:
            current_line_boxes.append(box)
            current_min_y = min(current_min_y, box.y)
            current_max_y = max(current_max_y, box.bottom)
        else:
            lines.append(TextLine(current_line_boxes))
            current_line_boxes = [box]
            current_min_y = box.y
            current_max_y = box.bottom
            
    if current_line_boxes:
        lines.append(TextLine(current_line_boxes))
        
    return lines

def segment_words(line: TextLine, space_threshold: float = None) -> list[list[BoundingBox]]:
    """
    Splits a line of boxes into words based on horizontal distance.
    """
    if not line.boxes:
        return []
        
    if space_threshold is None:
        # Estimate average character width to determine space threshold
        avg_width = sum(b.w for b in line.boxes) / len(line.boxes)
        space_threshold = avg_width * 1.5 # Heuristic
        
    words = []
    current_word = [line.boxes[0]]
    
    for box in line.boxes[1:]:
        prev_box = current_word[-1]
        distance = box.x - prev_box.right
        
        if distance > space_threshold:
            words.append(current_word)
            current_word = [box]
        else:
            current_word.append(box)
            
    if current_word:
        words.append(current_word)
        
    return words
