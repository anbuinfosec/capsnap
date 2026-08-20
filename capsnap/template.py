from .image import Image
from .features import normalize_character
import json
import math

class TemplateMatcher:
    def __init__(self):
        self.templates: dict[str, list[float]] = {}
        self.aspect_ratios: dict[str, float] = {}
        
    def add_template(self, char: str, image: Image):
        """Adds a template for a character."""
        norm = normalize_character(image, 28)
        
        # Calculate aspect ratio of original unpadded image bounds
        # But we only get the cropped image, so W / H
        aspect_ratio = image.width / max(1, image.height)
        
        features = [float(p > 127) for p in norm.pixels]
        
        self.templates[char] = features
        self.aspect_ratios[char] = aspect_ratio
        
    def _pixel_similarity(self, f1: list[float], f2: list[float]) -> float:
        if len(f1) != len(f2) or len(f1) == 0:
            return 0.0
        matches = sum(1 for a, b in zip(f1, f2) if a == b)
        return matches / len(f1)
        
    def _structural_similarity(self, f1: list[float], f2: list[float]) -> float:
        # A simple intersection over union of foreground pixels
        intersection = sum(1 for a, b in zip(f1, f2) if a == 1.0 and b == 1.0)
        union = sum(1 for a, b in zip(f1, f2) if a == 1.0 or b == 1.0)
        return intersection / union if union > 0 else 0.0
        
    def match(self, image: Image) -> tuple[str, float]:
        if not self.templates:
            return "?", 0.0
            
        norm = normalize_character(image, 28)
        features = [float(p > 127) for p in norm.pixels]
        aspect_ratio = image.width / max(1, image.height)
        
        best_char = "?"
        best_score = -1.0
        
        for char, temp_features in self.templates.items():
            pix_sim = self._pixel_similarity(features, temp_features)
            struct_sim = self._structural_similarity(features, temp_features)
            
            # Aspect ratio penalty
            ar_diff = abs(aspect_ratio - self.aspect_ratios.get(char, aspect_ratio))
            ar_penalty = max(0.0, 1.0 - ar_diff)
            
            # Weighted score
            score = (0.3 * pix_sim) + (0.5 * struct_sim) + (0.2 * ar_penalty)
            
            if score > best_score:
                best_score = score
                best_char = char
                
        return best_char, best_score
        
    def save(self, path: str):
        data = {
            "templates": self.templates,
            "aspect_ratios": self.aspect_ratios
        }
        with open(path, 'w') as f:
            json.dump(data, f)
            
    def load(self, path: str):
        with open(path, 'r') as f:
            data = json.load(f)
        self.templates = data.get("templates", {})
        self.aspect_ratios = data.get("aspect_ratios", {})
