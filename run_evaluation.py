#!/usr/bin/env python3
"""
Evaluate baseline DAM-3B outputs with standard captioning metrics.
"""

import json
import sys
import tempfile
import os
import pandas as pd
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap


def convert_predictions_to_coco_format(pred_dict):
    """Convert imgid_annid format to COCO results format."""
    results = []
    for key, caption in pred_dict.items():
        if '_' in key:
            img_id = int(key.split('_')[0])
            results.append({
                'image_id': img_id,
                'caption': caption
            })
    return results


def main():
    # File paths
    gt_file = 'annotations.json'
    pred_file = 'reconstructed_baseline_outputs.json'

    print("="*80)
    print("BASELINE EVALUATION - Standard Captioning Metrics")
    print("="*80)

    # Step 1: Load predictions
    print(f"\n[1/4] Loading predictions from {pred_file}...")
    with open(pred_file, 'r') as f:
        pred_dict = json.load(f)

    predictions = convert_predictions_to_coco_format(pred_dict)
    print(f"      ✓ Loaded {len(predictions)} predictions")

    # Step 2: Load ground truth
    print(f"\n[2/4] Loading ground truth from {gt_file}...")
    coco = COCO(gt_file)

    # Add missing 'info' field if not present
    if 'info' not in coco.dataset:
        coco.dataset['info'] = {'description': 'DLC-Bench'}
    if 'licenses' not in coco.dataset:
        coco.dataset['licenses'] = []

    print(f"      ✓ Loaded {len(coco.getImgIds())} images")

    # Step 3: Prepare evaluation
    print(f"\n[3/4] Preparing evaluation...")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(predictions, f)
        temp_pred_file = f.name

    coco_result = coco.loadRes(temp_pred_file)
    coco_eval = COCOEvalCap(coco, coco_result)

    # Step 4: Run evaluation
    print(f"\n[4/4] Computing metrics...")
    print("      (SPICE may take 2-3 minutes...)")
    coco_eval.evaluate()

    # Display results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)

    results = {}
    for metric, score in coco_eval.eval.items():
        results[metric] = score
        print(f"  {metric:12s}: {score:.4f}")

    print("="*80)

    # Create DataFrame for better visualization
    df = pd.DataFrame([results])

    # Reorder columns
    column_order = ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4',
                    'METEOR', 'ROUGE_L', 'CIDEr', 'SPICE']
    available_cols = [col for col in column_order if col in df.columns]
    df = df[available_cols]

    print("\n\nFormatted Results Table:")
    print("="*80)
    print(df.to_string(index=False, float_format='%.4f'))
    print("="*80)

    # Save results
    results_json = 'baseline_metrics_results.json'
    results_csv = 'baseline_metrics_results.csv'

    with open(results_json, 'w') as f:
        json.dump(results, f, indent=2)

    df.to_csv(results_csv, index=False)

    print(f"\n✓ Results saved:")
    print(f"  - JSON: {results_json}")
    print(f"  - CSV:  {results_csv}")

    # Cleanup
    os.unlink(temp_pred_file)

    print("\nEvaluation complete!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
