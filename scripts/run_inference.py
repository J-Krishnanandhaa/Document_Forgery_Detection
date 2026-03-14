"""
Inference Script for Document Forgery Detection

Run inference on single images or entire directories.

Usage:
    python scripts/run_inference.py --input path/to/image.jpg --model outputs/checkpoints/best_doctamper.pth
    python scripts/run_inference.py --input path/to/folder/ --model outputs/checkpoints/best_doctamper.pth
"""

import argparse
import sys
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config
from src.inference import get_pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Run forgery detection inference")
    
    parser.add_argument('--input', type=str, required=True,
                       help='Input image or directory path')
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to localization model checkpoint')
    
    parser.add_argument('--classifier', type=str, default=None,
                       help='Path to classifier directory (optional)')
    
    parser.add_argument('--output', type=str, default='outputs/results',
                       help='Output directory')
    
    parser.add_argument('--is_text', action='store_true',
                       help='Enable OCR features for text documents')
    
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    
    return parser.parse_args()


def process_file(pipeline, input_path: str, output_dir: str):
    """Process a single file"""
    try:
        result = pipeline.run(input_path, output_dir)
        return result
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return None


def main():
    args = parse_args()
    
    # Load config
    config = get_config(args.config)
    
    print("\n" + "="*60)
    print("Hybrid Document Forgery Detection - Inference")
    print("="*60)
    print(f"Input: {args.input}")
    print(f"Model: {args.model}")
    print(f"Classifier: {args.classifier or 'None'}")
    print(f"Output: {args.output}")
    print("="*60)
    
    # Create pipeline
    pipeline = get_pipeline(
        config,
        model_path=args.model,
        classifier_path=args.classifier,
        is_text_document=args.is_text
    )
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get input files
    input_path = Path(args.input)
    
    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        extensions = ['.jpg', '.jpeg', '.png', '.pdf', '.bmp', '.tiff']
        files = [f for f in input_path.iterdir() 
                if f.suffix.lower() in extensions]
    else:
        print(f"Invalid input path: {input_path}")
        return
    
    print(f"\nProcessing {len(files)} file(s)...")
    
    # Process files
    all_results = []
    
    for file_path in files:
        result = process_file(pipeline, str(file_path), str(output_dir))
        if result:
            all_results.append(result)
            
            # Print summary
            status = "TAMPERED" if result['is_tampered'] else "AUTHENTIC"
            print(f"\n  {file_path.name}: {status}")
            if result['is_tampered']:
                print(f"    Regions detected: {result['num_regions']}")
                for region in result['regions'][:3]:  # Show first 3
                    print(f"    - {region['forgery_type']} (conf: {region['confidence']:.2f})")
    
    # Save summary
    summary_path = output_dir / 'inference_summary.json'
    summary = {
        'total_files': len(files),
        'processed': len(all_results),
        'tampered': sum(1 for r in all_results if r['is_tampered']),
        'authentic': sum(1 for r in all_results if not r['is_tampered']),
        'results': all_results
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print("\n" + "="*60)
    print("Inference Complete!")
    print(f"Total: {summary['total_files']}, "
          f"Tampered: {summary['tampered']}, "
          f"Authentic: {summary['authentic']}")
    print(f"Results saved to: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
