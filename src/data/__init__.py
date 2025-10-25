"""
Data module for Hierarchical InLegalBERT.
Contains dataset loading and processing utilities.
"""

from .dataset import ILDCDataset, load_ildc_dataset

__all__ = ["ILDCDataset", "load_ildc_dataset"]
