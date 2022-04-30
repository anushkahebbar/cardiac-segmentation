# Semantic Segmentation of Anatomical ROIs in Chest CT Scans


### Directory Structure
- The first folder, `1 - Methods with Improved Results` contains files with code presenting our improved workflow for lung, trachea, spine and heart segmentation. It results in more accurate segmentation of meaningful anatomical structures.
- The second folder, `2 - KMeans & Morphology Methods - Rim et al.` is a complete, original implementation of the paper by Rim et al., 'Semantic Cardiac Segmentation in Chest CT Images Using K-Means Clustering and the Mathematical Morphology Method'.
- The third folder, `3 - Statistical Parameter Methods - Larrey-Ruiz et al.` implements the paper by Larrey-Ruiz et al., 'Automatic image-based segmentation of the heart from CT scans', and is heavily derived from the [GitHub repo by @karageorge](https://github.com/karageorge/Automatic-image-based-segmentation-of-the-heart-from-CTs). This was mainly done to explore different techniques of dealing with CT scan images. None of the ideas from the code in this folder are implemented in our original work in folder 1.
```
.
├── 1 - Methods with Improved Results
│   ├── anatomical_roi_segmentation_model.ipynb
│   ├── model_pipeline.ipynb
│   └── SegmentationFunctions.py
├── 2 - KMeans & Morphology Methods - Rim et al.
│   ├── Histograms with KMeans
│   │   ├── histograms_with_clusters_and_threshold.ipynb
│   │   ├── rough_draft_histograms.ipynb
│   │   └── rough_draft_hist_with_kmeans.ipynb
│   ├── model_pipeline.ipynb
│   ├── SegmentationFunctions.py
│   ├── segmentation_model_on_sample#16_truePositive.ipynb
│   └── segmentation_model_on_sample#2_falsePositive.ipynb
├── 3 - Statistical Parameter Methods - Larrey-Ruiz et al.
│   └── preprocessing_with_hist_params.ipynb
├── LICENSE
├── pipeline-imgs-report
├── README.md
├── sample-dataset
└── sample-dataset-ground-truth
```
