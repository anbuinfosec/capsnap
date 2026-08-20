import os
import json
from .image import Image
from .features import normalize_character

def read_pgm(path: str) -> Image:
    """Reads a binary PGM file."""
    with open(path, 'rb') as f:
        header = f.readline().decode('ascii').strip()
        if header != 'P5':
            raise ValueError(f"Invalid PGM header in {path}")
            
        # skip comments
        while True:
            pos = f.tell()
            line = f.readline().decode('ascii')
            if not line.startswith('#'):
                f.seek(pos)
                break
                
        dims = f.readline().decode('ascii').strip().split()
        width = int(dims[0])
        height = int(dims[1])
        
        max_val = int(f.readline().decode('ascii').strip())
        
        pixels = f.read(width * height)
        
    return Image(width, height, 1, pixels)

def load_dataset(dataset_dir: str, split: str = "train", target_size: int = 28) -> tuple[list[list[float]], list[str]]:
    """
    Loads images from a dataset split and returns (features, labels).
    """
    labels_file = os.path.join(dataset_dir, "labels.json")
    with open(labels_file, 'r') as f:
        labels_dict = json.load(f)[split]
        
    split_dir = os.path.join(dataset_dir, split)
    
    X = []
    y = []
    
    for filename, char in labels_dict.items():
        path = os.path.join(split_dir, filename)
        image = read_pgm(path)
        
        # Invert to make text=255, background=0
        for i in range(len(image.pixels)):
            image.pixels[i] = 255 - image.pixels[i]
            
        norm = normalize_character(image, target_size)
        features = [float(p > 127) for p in norm.pixels]
        
        X.append(features)
        y.append(char)
        
    return X, y
