#  Segment Augmentation and Differentiable Ranking for Logo Retrieval

Code for ICPR2022 paper "Segment Augmentation and Differentiable Ranking for Logo Retrieval".
> [Segment Augmentation and Differentiable Ranking for Logo Retrieval](https://arxiv.org/abs/2209.02482)
## Abstract
Logo retrieval is a challenging problem since the definition of similarity is more subjective compared to image retrieval tasks and the set of known similarities is very scarce. To tackle this challenge, in this paper, we propose a simple but effective segment-based augmentation strategy to introduce artificially similar logos for training deep networks for logo retrieval. In this novel augmentation strategy, we first find segments in a logo and apply transformations such as rotation, scaling, and color change, on the segments, unlike the conventional image-level augmentation strategies. Moreover, we evaluate whether the recently introduced ranking-based loss function, Smooth-AP, is a better approach for learning similarity for logo retrieval. On the large scale METU Trademark Dataset, we show that (i) our segment-based augmentation strategy improves retrieval performance compared to the baseline model or image-level augmentation strategies, and (ii) Smooth-AP indeed performs better than conventional losses for logo retrieval. 
<p align="center">
<img src=https://github.com/blisgard/SegmentBasedAugmentation/blob/main/figures/teaser_figures.png  width="450" height="300">
</p>

## Method

We perform segment level augmentation by following these steps:
1. **Logo segmentation** 
2. **Segment selection** 
3. **Segment transformation**

<p align="center" >
  <img src=https://github.com/blisgard/SegmentBasedAugmentation/blob/main/figures/sample_augmentations.png  width="300" height="360">
</p>

Currently, our segment-level augmentation consists 3 methods: Color change, Segment Removal and Segment Rotation. For details, please visit the paper. 


## Presentation Video

You can access the short version of paper presentation from [here](https://www.youtube.com/watch?v=06d4OvMqmWg). 

## Experimental Results

**The Effect of Image-Level (H. Flip, V. Flip) and Segment-Level
Augmentation**:

| Method        | NAR           | 
| ------------- |:-------------:| 
| Baseline(No augmentation)    |  0.102 | 
| Triplet Loss(No augmentation)      | 0.053      | 
| Triplet Loss(Image-level aug.)|  0.051      |  
| Triplet Loss(S. Color, S. Removal)|  0.046     |  
| Smooth-AP Loss(No augmentation)|  0.046      |  
| Smooth-AP Loss(Image-level aug.)|  0.044      |  
| Smooth-AP Loss(S. Color) |  0.040      |  

## Visual Results
![](https://github.com/blisgard/SegmentBasedAugmentation/blob/main/figures/visual_results_1.png)
![](https://github.com/blisgard/SegmentBasedAugmentation/blob/main/figures/visual_results_2.png)

## Dataset
In this repository, [METU Trademark Dataset](https://github.com/neouyghur/METU-TRADEMARK-DATASET) is used. The dataset is available per request only. If you are a researcher at a university (or a graduate student) interested in the dataset for research purposes, please contact Sinan Kalkan with your intention.
## Dependencies

Dependencies can be found in requirements.txt. 

Use `pip3 install requirements.txt`


## Training the model

python3 Standard_Training.py  --bs 512 --lr 0.001 --n_epochs 50


