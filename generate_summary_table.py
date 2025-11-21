#!/usr/bin/env python3
"""
Generate a summary table of reconstructed baseline outputs.
"""

import json
import pandas as pd
from pycocotools.coco import COCO


def main():
    print("="*80)
    print("BASELINE OUTPUTS SUMMARY")
    print("="*80)

    # Load predictions
    print("\nLoading reconstructed baseline outputs...")
    with open('reconstructed_baseline_outputs.json', 'r') as f:
        predictions = json.load(f)

    print(f"✓ Loaded {len(predictions)} predictions")

    # Load annotations
    print("\nLoading DLC-Bench annotations...")
    coco = COCO('annotations.json')
    print(f"✓ Loaded {len(coco.getImgIds())} images with {len(coco.getAnnIds())} annotations")

    # Analyze predictions
    caption_lengths = [len(cap.split()) for cap in predictions.values()]
    avg_length = sum(caption_lengths) / len(caption_lengths)
    min_length = min(caption_lengths)
    max_length = max(caption_lengths)

    # Create summary statistics
    print("\n" + "="*80)
    print("CAPTION STATISTICS")
    print("="*80)
    print(f"  Total predictions:        {len(predictions)}")
    print(f"  Average caption length:   {avg_length:.1f} words")
    print(f"  Min caption length:       {min_length} words")
    print(f"  Max caption length:       {max_length} words")
    print("="*80)

    # Sample predictions
    print("\n" + "="*80)
    print("SAMPLE PREDICTIONS")
    print("="*80)

    sample_keys = list(predictions.keys())[:5]
    for key in sample_keys:
        img_id, ann_id = key.split('_')
        caption = predictions[key]
        print(f"\nImage {img_id}, Annotation {ann_id}:")
        print(f"  {caption[:150]}{'...' if len(caption) > 150 else ''}")

    print("\n" + "="*80)

    # Create summary table
    summary_data = {
        'Metric': [
            'Total Predictions',
            'Unique Images',
            'Avg Caption Length (words)',
            'Min Caption Length (words)',
            'Max Caption Length (words)'
        ],
        'Value': [
            len(predictions),
            len(set(k.split('_')[0] for k in predictions.keys())),
            f"{avg_length:.1f}",
            min_length,
            max_length
        ]
    }

    df = pd.DataFrame(summary_data)

    print("\nSUMMARY TABLE")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)

    # Save summary
    df.to_csv('baseline_summary.csv', index=False)
    print(f"\n✓ Summary saved to: baseline_summary.csv")

    # Note about evaluation
    print("\n" + "="*80)
    print("NOTE: DLC-Bench Evaluation")
    print("="*80)
    print("DLC-Bench uses LLM judge evaluation (not BLEU/CIDEr/METEOR).")
    print("To run evaluation, use:")
    print("  python evaluation/eval_model_outputs.py --pred <predictions.json>")
    print("\nStandard metrics (BLEU, CIDEr, etc.) require reference captions,")
    print("which are not provided in DLC-Bench.")
    print("="*80)


if __name__ == '__main__':
    main()
