from .image import Image
from .components import BoundingBox
from .features import extract_patch, normalize_character
from .model import MLP
from .template import TemplateMatcher
import os

class Recognizer:
    def __init__(self, model_path: str = None, template_path: str = None):
        self.model = None
        self.classes = []
        self.template_matcher = None
        
        if model_path and os.path.exists(model_path):
            self.model, self.classes = MLP.load(model_path)
            
        if template_path and os.path.exists(template_path):
            self.template_matcher = TemplateMatcher()
            self.template_matcher.load(template_path)
            
    def _image_to_features(self, image: Image) -> list[float]:
        # Convert binary image (0/255) to 0.0/1.0 float list
        return [float(p > 127) for p in image.pixels]
        
    def recognize_character(self, image: Image) -> tuple[str, float]:
        """
        Takes a tightly cropped binary character image, normalizes it, and predicts.
        Returns (character, confidence)
        """
        if self.model:
            import math
            target_size = int(math.sqrt(self.model.layer_sizes[0]))
            norm = normalize_character(image, target_size)
            features = self._image_to_features(norm)
            preds, _, _ = self.model.forward(features)
            best_idx = preds.index(max(preds))
            return self.classes[best_idx], preds[best_idx]
        elif self.template_matcher:
            return self.template_matcher.match(image)
        else:
            return "?", 0.0

    def recognize_line(self, original_bin: Image, words: list[list[BoundingBox]]) -> str:
        text = []
        for word in words:
            word_str = ""
            for box in word:
                patch = extract_patch(original_bin, box)
                char, conf = self.recognize_character(patch)
                word_str += char
            text.append(word_str)
        return " ".join(text)
