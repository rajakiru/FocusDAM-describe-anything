#!/usr/bin/env python3
"""
Evaluate baseline DAM-3B outputs using standard captioning metrics.
Computes BLEU, CIDEr, METEOR, ROUGE-L, and SPICE scores.
"""

import json
import sys
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap


def load_predictions(pred_file):
    """Load predictions from JSON file."""
    print(f"Loading predictions from {pred_file}...")
    with open(pred_file, 'r') as f:
        predictions = json.load(f)

    # Convert to COCO format: list of dicts with 'image_id' and 'caption'
    results = []
    for key, caption in predictions.items():
        # Key format: "imgid_annid"
        if '_' in key:
            img_id, ann_id = key.split('_')[0:2]
            results.append({
                'image_id': int(img_id),
                'caption': caption
            })

    print(f"Loaded {len(results)} predictions")
    return results


def evaluate_captions(gt_file, pred_file, output_file=None):
    """
    Evaluate captions using standard metrics.

    Args:
        gt_file: Path to DLC-Bench annotations.json
        pred_file: Path to predictions JSON file
        output_file: Optional path to save results JSON
    """
    # Load ground truth annotations
    print(f"\nLoading ground truth from {gt_file}...")
    coco = COCO(gt_file)

    # Load predictions
    predictions = load_predictions(pred_file)

    # Create temporary file for COCO format predictions
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(predictions, f)
        temp_pred_file = f.name

    # Load predictions into COCO format
    print("\nPreparing evaluation...")
    coco_result = coco.loadRes(temp_pred_file)

    # Create COCOEval object
    coco_eval = COCOEvalCap(coco, coco_result)

    # Evaluate
    print("\nComputing metrics...")
    coco_eval.evaluate()

    # Print results
    print("\n" + "="*80)
    print("BASELINE EVALUATION RESULTS (Standard Captioning Metrics)")
    print("="*80)

    results = {}
    for metric, score in coco_eval.eval.items():
        print(f"{metric:12s}: {score:.4f}")
        results[metric] = float(score)

    print("="*80)

    # Detailed scores per image (optional)
    print("\nPer-image scores computed. Use coco_eval.evalImgs for details.")

    # Save results if output file specified
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")

    # Cleanup temp file
    import os
    os.unlink(temp_pred_file)

    return results


def main():
    """Main evaluation function."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Evaluate baseline outputs with standard captioning metrics'
    )
    parser.add_argument(
        '--gt',
        type=str,
        default='evaluation/DLC-bench/annotations.json',
        help='Path to ground truth annotations (DLC-Bench annotations.json)'
    )
    parser.add_argument(
        '--pred',
        type=str,
        required=True,
        help='Path to predictions JSON file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Optional path to save results JSON'
    )

    args = parser.parse_args()

    try:
        results = evaluate_captions(args.gt, args.pred, args.output)
        return 0
    except Exception as e:
        print(f"\nError during evaluation: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
