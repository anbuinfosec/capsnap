import argparse
import sys
import json
from .api import OCR

def main():
    parser = argparse.ArgumentParser(description="capsnap - A pure-Python zero-dependency OCR engine")
    parser.add_argument("image", type=str, help="Path to the PNG image")
    parser.add_argument("--json", action="store_true", help="Output result as JSON")
    parser.add_argument("--confidence", action="store_true", help="Output confidence score")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (saves intermediate images)")
    parser.add_argument("--language", type=str, default="eng", help="Language code")
    parser.add_argument("--model", type=str, help="Path to custom model (.capsnap)")
    parser.add_argument("--template", type=str, help="Path to custom templates (template.json)")
    
    args = parser.parse_args()
    
    ocr = OCR(
        language=args.language,
        debug=args.debug,
        model_path=args.model,
        template_path=args.template
    )
    
    try:
        result = ocr.read(args.image)
        
        if args.json:
            out = {
                "text": result.text,
                "confidence": result.confidence
            }
            print(json.dumps(out, indent=2))
        else:
            print(result.text)
            if args.confidence:
                print(f"Confidence: {result.confidence:.4f}")
                
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
