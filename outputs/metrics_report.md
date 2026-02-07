# Perception Pipeline Metrics Report

**Generated:** 2026-02-03 21:59:13

## Executive Summary

| Metric | Value |
|--------|-------|
| mAP@0.5 (2D) | 0.576 |
| 2D Precision | 0.512 |
| 2D Recall | 0.977 |
| 3D Precision | 0.360 |
| 3D Recall | 0.484 |
| Mean Center Error | 1.42m |
| Mean 3D IoU | 0.085 |

## 2D Detection Metrics

### Overall Performance

| Metric | Value |
|--------|-------|
| mAP@0.5 | 0.576 |
| Precision | 0.512 |
| Recall | 0.977 |
| F1 Score | 0.672 |

### Per-Class Performance

| Class | AP | Precision | Recall | F1 | TP | FP | GT |
|-------|-----|-----------|--------|-----|-----|-----|-----|
| Car | 0.750 | 0.524 | 1.000 | 0.687 | 33 | 30 | 33 |
| Pedestrian | 0.978 | 0.900 | 1.000 | 0.947 | 9 | 1 | 9 |
| Cyclist | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 9 | 1 |

## 3D Detection Metrics

### Error Metrics

| Metric | Value |
|--------|-------|
| Mean Center Error | 1.415m |
| Median Center Error | 1.373m |
| Mean Orientation Error | 85.5° |
| Mean 3D IoU | 0.085 |
| Dimension Error (L) | 11.424m |
| Dimension Error (W) | 1.256m |
| Dimension Error (H) | 0.210m |

### Accuracy by Distance

| Distance | Count | Mean Center Error | Mean 3D IoU |
|----------|-------|-------------------|-------------|
| 0-20m | 12 | 1.513m | 0.075 |
| 20-40m | 12 | 1.382m | 0.062 |
| 40-80m | 7 | 1.305m | 0.142 |

## Metric Definitions

### 2D Metrics

- **Precision**: TP / (TP + FP) - Fraction of detections that are correct
- **Recall**: TP / (TP + FN) - Fraction of ground truths that are detected
- **mAP@0.5**: Mean Average Precision at IoU threshold 0.5
- **AP**: Area under the Precision-Recall curve

### 3D Metrics

- **Center Distance Error**: Euclidean distance |predicted_center - GT_center| in meters
- **Orientation Error**: Angular difference in yaw, wrapped to [0, π]
- **3D IoU**: Intersection over Union of 3D bounding boxes
- **Dimension Error**: |predicted_dimensions - GT_dimensions| for L/W/H

### Distance Bins

- **Near (0-20m)**: Close-range objects, typically highest accuracy
- **Medium (20-40m)**: Mid-range objects
- **Far (40m+)**: Distant objects, challenging due to sparse LiDAR points

## Interpretation Guide

### Quality Thresholds

| Metric | Good | Acceptable | Poor |
|--------|------|------------|------|
| mAP@0.5 | > 0.7 | 0.5-0.7 | < 0.5 |
| 3D IoU | > 0.5 | 0.3-0.5 | < 0.3 |
| Center Error | < 1m | 1-2m | > 2m |
| Orientation Error | < 10° | 10-30° | > 30° |
