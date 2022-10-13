# CMRxMotion Task 1
Code example for MICCAI2022 challenge CMRxMotion task 1 - Cardiac MRI motion artefacts classification


Paper: Motion-related Artefact Classification Using Patch-based Ensemble and Transfer Learning in Cardiac MRI


Challenge website: http://cmr.miccai.cloud/

## Usage
### Requirements
Python 3.9 \
PyTorch 1.11
```bash
pip install -r requirement.txt
```
### Model Training
1. Train heart region segmentation model (`train_seg.py`) on <a href="[URL](https://www.synapse.org/#!Synapse:syn32407769/wiki/618236)">CMRxMotion task 2 data</a>.
2. Crop the 3D scans into 2D slices and segment the heart region assisted by the segmentation model (`pred_crop.py`).
3. Divide the slices into 5-fold for cross-validation for classification model training (`split_slices.py`).
4. Train classification models (`train_seg.py`, `train_seg.py`, `train_seg.py`) on 2D slices.
   
### Model inference
Crop the validation data into 2D slices and segment the heart region (`pred_crop.py`), then run `inference.py`.


