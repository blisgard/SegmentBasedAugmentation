#  Segment Augmentation and Differentiable Ranking for Logo Retrieval

Code for ICPR2022 paper "Segment Augmentation and Differentiable Ranking for Logo Retrieval".
> [Segment Augmentation and Differentiable Ranking for Logo Retrieval](https://arxiv.org/abs/2209.02482)
## Abstract
Logo retrieval is a challenging problem since the definition of similarity is more subjective compared to image retrieval tasks and the set of known similarities is very scarce. To tackle this challenge, in this paper, we propose a simple but effective segment-based augmentation strategy to introduce artificially similar logos for training deep networks for logo retrieval. In this novel augmentation strategy, we first find segments in a logo and apply transformations such as rotation, scaling, and color change, on the segments, unlike the conventional image-level augmentation strategies. Moreover, we evaluate whether the recently introduced ranking-based loss function, Smooth-AP, is a better approach for learning similarity for logo retrieval. On the large scale METU Trademark Dataset, we show that (i) our segment-based augmentation strategy improves retrieval performance compared to the baseline model or image-level augmentation strategies, and (ii) Smooth-AP indeed performs better than conventional losses for logo retrieval. 

## Method
![](https://github.com/blisgard/SegmentBasedAugmentation/blob/main/figures/sample_augmentations.png)
![](https://github.com/blisgard/SegmentBasedAugmentation/blob/main/figures/rotation_steps.png)
![](https://github.com/blisgard/SegmentBasedAugmentation/blob/main/figures/teaser_figures.png)
## Presentation Video

You can access the short version of paper presentation from [here](https://www.youtube.com/watch?v=06d4OvMqmWg). 

## Visual Results
![](https://github.com/blisgard/SegmentBasedAugmentation/blob/main/figures/visual_results_1.png)
![](https://github.com/blisgard/SegmentBasedAugmentation/blob/main/figures/visual_results_2.png)

## Dataset
In this repository, [METU Trademark Dataset](https://github.com/neouyghur/METU-TRADEMARK-DATASET) is used. The dataset is available per request only. If you are a researcher at a university (or a graduate student) interested in the dataset for research purposes, please contact Sinan Kalkan with your intention.
## Dependencies





## Training the model

python3 Standard_Training.py  --bs 512 --lr 0.001 --n_epochs 50


