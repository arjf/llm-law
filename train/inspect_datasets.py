#!/usr/bin/env python3
"""
Inspect IL-TUR dataset schemas to understand actual field names
This helps us write correct data formatting functions
"""

from datasets import load_dataset
import json


def inspect_dataset(name, subset, split):
    """Load and inspect a dataset's schema"""
    print(f"\n{'=' * 70}")
    print(f"Dataset: {name} | Subset: {subset} | Split: {split}")
    print(f"{'=' * 70}")

    try:
        ds = load_dataset(name, subset, split=split)
        print(f"✓ Loaded successfully: {len(ds)} samples")
        print(f"\nColumn names: {ds.column_names}")

        if len(ds) > 0:
            print(f"\nFirst item fields:")
            first_item = ds[0]
            for key, value in first_item.items():
                value_preview = (
                    str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                )
                value_type = type(value).__name__
                print(f"  - {key} ({value_type}): {value_preview}")

            # Show full first item as JSON
            print(f"\nFull first item (JSON):")
            print(json.dumps(first_item, indent=2, default=str)[:1000] + "...")
        else:
            print("⚠ Dataset is empty")

    except Exception as e:
        print(f"✗ Failed to load: {e}")


def main():
    print("=" * 70)
    print("IL-TUR Dataset Schema Inspector")
    print("=" * 70)

    # Inspect the 3 training tasks
    datasets_to_inspect = [
        ("Exploration-Lab/IL-TUR", "pcr", "train_queries"),
        ("Exploration-Lab/IL-TUR", "lsi", "train"),
        ("Exploration-Lab/IL-TUR", "summ", "train"),
    ]

    for name, subset, split in datasets_to_inspect:
        inspect_dataset(name, subset, split)

    print("\n" + "=" * 70)
    print("Inspection complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Check which fields actually exist in each dataset")
    print("2. Update the _format_* functions in rag.py accordingly")
    print("3. Ensure training data has proper labels/targets")


if __name__ == "__main__":
    main()
