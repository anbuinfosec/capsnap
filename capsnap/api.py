import base64
import re
import os

from .image import Image
from .decoder import read_png, write_png
from .grayscale import to_grayscale
from .threshold import threshold_otsu
from .morphology import opening
from .components import find_components, BoundingBox
from .segmentation import group_into_lines, segment_words, TextLine
from .recognize import Recognizer

def _looks_like_base64(s: str) -> bool:
    """Heuristic: if the string is long and contains only base64 chars, treat it as base64."""
    import string
    if len(s) < 32:
        return False
    allowed = set(string.ascii_letters + string.digits + '+/=\n\r ')
    sample = s[:256]
    return all(c in allowed for c in sample)

class OCRResult:
    def __init__(self, text: str, confidence: float, lines: list):
        self.text = text
        self.confidence = confidence
        self.lines = lines
        
class OCRLine:
    def __init__(self, text: str, confidence: float, words: list):
        self.text = text
        self.confidence = confidence
        self.words = words
        
class OCRWord:
    def __init__(self, text: str, confidence: float, characters: list):
        self.text = text
        self.confidence = confidence
        self.characters = characters

class OCR:
    def __init__(self, language: str = "eng", mode: str = "document", preprocess: bool = True, model_path: str = None, template_path: str = None, debug: bool = False):
        self.language = language
        self.mode = mode
        self.preprocess = preprocess
        self.debug = debug
        
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), "model.capsnap")
        if template_path is None:
            template_path = os.path.join(os.path.dirname(__file__), "template.json")
            
        self.recognizer = Recognizer(model_path, template_path)
        
    def read(self, source) -> 'OCRResult':
        """
        Universal entry point. Accepts:
          - str / os.PathLike  → file path on disk
          - bytes              → raw PNG/JPEG bytes
          - str (base64)       → base64-encoded image with or without data-URL prefix
        """
        if isinstance(source, (bytes, bytearray)):
            return self.read_bytes(bytes(source))

        if isinstance(source, os.PathLike):
            source = str(source)

        if isinstance(source, str):
            # data-URL or plain base64 string
            if source.startswith('data:') or _looks_like_base64(source):
                return self.read_base64(source)
            # Otherwise treat as a file path
            return self.read_path(source)

        raise TypeError(
            f"read() expects a file path (str/Path), bytes, or base64 string; "
            f"got {type(source).__name__!r}"
        )

    def read_path(self, path: str) -> 'OCRResult':
        """Read from a file path on disk."""
        with open(path, 'rb') as f:
            data = f.read()
        return self.read_bytes(data)

    def read_base64(self, b64_str: str) -> 'OCRResult':
        """Read from a base64 string (with or without data-URL prefix)."""
        # Strip data URL prefix: data:image/png;base64,  or data:image/jpeg;base64,
        match = re.match(r'^data:image/[^;]+;base64,(.+)$', b64_str, re.DOTALL)
        if match:
            b64_str = match.group(1)

        # Fix missing base64 padding
        b64_str = b64_str.strip()
        padding = (4 - len(b64_str) % 4) % 4
        b64_str += '=' * padding

        data = base64.b64decode(b64_str)
        return self.read_bytes(data)

    def read_bytes(self, data: bytes) -> 'OCRResult':
        """Read from raw image bytes (PNG)."""
        image = read_png(data)
        if self.debug:
            self._save_debug("debug_original.png", image)
        return self._process_image(image)
        
    def _save_debug(self, filename: str, image: Image):
        try:
            with open(filename, 'wb') as f:
                f.write(write_png(image))
        except Exception as e:
            print(f"Failed to save debug image {filename}: {e}")
            
    def _process_image(self, image: Image) -> OCRResult:
        # 1. Grayscale
        gray = to_grayscale(image)
        if self.debug:
            self._save_debug("debug_grayscale.png", gray)
            
        # 2. Threshold (invert to make text=255)
        binary = threshold_otsu(gray, invert=True)
        if self.debug:
            self._save_debug("debug_threshold.png", binary)
            
        # 3. Morphology (cleanup noise)
        if self.preprocess:
            binary = opening(binary, kernel_size=3)
            if self.debug:
                self._save_debug("debug_morphology.png", binary)
                
        # 4. Segmentation
        if self.mode == "captcha":
            boxes = []
            for i in range(5):
                x_start = int(i * binary.width / 5)
                x_end = int((i + 1) * binary.width / 5)
                slice_w = x_end - x_start
                
                # Copy slice pixels
                slice_pixels = bytearray(slice_w * binary.height)
                for y in range(binary.height):
                    for x in range(slice_w):
                        slice_pixels[y * slice_w + x] = binary.pixels[y * binary.width + x_start + x]
                
                slice_img = Image(slice_w, binary.height, 1, slice_pixels)
                slice_boxes = find_components(slice_img)
                
                if slice_boxes:
                    largest_box = max(slice_boxes, key=lambda b: b.w * b.h)
                    
                    # Adjust box coordinates to global image
                    box = BoundingBox(
                        x_start + largest_box.x,
                        largest_box.y,
                        largest_box.w,
                        largest_box.h
                    )
                    boxes.append(box)
        else:
            boxes = find_components(binary)
            # Filter extremely small boxes (noise)
            boxes = [b for b in boxes if b.w > 2 and b.h > 2]
        
        if self.mode == "captcha":
            # For CAPTCHAs, we already have exactly 4 boxes (or fewer if some slices were empty).
            # We don't want to group them into lines or words based on distance heuristics, 
            # as that might split them incorrectly.
            # Just force them all into a single line and a single word.
            lines = [TextLine(boxes)] if boxes else []
            
            result_lines = []
            overall_conf = 0.0
            char_count = 0
            
            if lines:
                line = lines[0]
                ocr_words = []
                word_chars = []
                word_conf = 0.0
                
                # Sort boxes by X to read left to right
                sorted_boxes = sorted(line.boxes, key=lambda b: b.x)
                
                for box in sorted_boxes:
                    from .features import extract_patch
                    patch = extract_patch(binary, box)
                    char, conf = self.recognizer.recognize_character(patch)
                    
                    word_chars.append({"char": char, "box": box, "confidence": conf})
                    word_conf += conf
                
                avg_word_conf = word_conf / len(sorted_boxes) if sorted_boxes else 0.0
                word_text = "".join(c["char"] for c in word_chars)
                ocr_words.append(OCRWord(word_text, avg_word_conf, word_chars))
                
                line_text = word_text
                result_lines.append(OCRLine(line_text, avg_word_conf, ocr_words))
                
                overall_conf = word_conf
                char_count = len(sorted_boxes)
                
            final_conf = overall_conf / char_count if char_count > 0 else 0.0
            final_text = "\n".join(l.text for l in result_lines)
            
            return OCRResult(final_text, final_conf, result_lines)
        
        lines = group_into_lines(boxes)
        
        # 5. Recognition
        result_lines = []
        overall_conf = 0.0
        char_count = 0
        
        for line in lines:
            words = segment_words(line)
            ocr_words = []
            
            line_conf = 0.0
            line_chars = 0
            
            for word_boxes in words:
                word_chars = []
                word_conf = 0.0
                
                for box in word_boxes:
                    from .features import extract_patch
                    patch = extract_patch(binary, box)
                    char, conf = self.recognizer.recognize_character(patch)
                    
                    word_chars.append({"char": char, "box": box, "confidence": conf})
                    word_conf += conf
                    
                avg_word_conf = word_conf / len(word_boxes) if word_boxes else 0.0
                word_text = "".join(c["char"] for c in word_chars)
                ocr_words.append(OCRWord(word_text, avg_word_conf, word_chars))
                
                line_conf += word_conf
                line_chars += len(word_boxes)
                
            avg_line_conf = line_conf / line_chars if line_chars > 0 else 0.0
            line_text = " ".join(w.text for w in ocr_words)
            result_lines.append(OCRLine(line_text, avg_line_conf, ocr_words))
            
            overall_conf += line_conf
            char_count += line_chars
            
        final_conf = overall_conf / char_count if char_count > 0 else 0.0
        final_text = "\n".join(l.text for l in result_lines)
        
        return OCRResult(final_text, final_conf, result_lines)
