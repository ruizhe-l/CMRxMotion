# CMRxMotion Task 1
Code example for MICCAI2022 challenge CMRxMotion task 1 - Cardiac MRI motion artefacts classification \\
Paper: Motion-related Artefacts Classification Using Patch-based Ensemble and Transfer Learning in Cardiac MRI \\
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
2. Segment and crop the 3D scan into 2D slices assited by the segmentation model the data (`pred_crop.py`).
3. Divide the slices into 5-fold for 5-fold cross validation on classification models training (`split_slices.py`).
4. Train classification models (`train_seg.py`, `train_seg.py`, `train_seg.py`) region segmentation model on 2D slices.
   
### Model inference
Segment and crop the validation data first (`pred_crop.py`), then run (`inference.py`)


