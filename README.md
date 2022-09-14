#  Segment Augmentation and Differentiable Ranking for Logo Retrieval

Officiel code for our [ICPR2022](https://www.icpr2022.com/) paper: 

Feyza Yavuz and [Sinan Kalkan](https://user.ceng.metu.edu.tr/~skalkan/), "[Segment Augmentation and Differentiable Ranking for Logo Retrieval](https://arxiv.org/abs/2209.02482)", 26th International Conference on Pattern Recognition (ICPR), 2022. [[Arxiv](https://arxiv.org/abs/2209.02482)] [[Video](https://www.youtube.com/watch?v=06d4OvMqmWg)] 

## Abstract
Logo retrieval is a challenging problem since the definition of similarity is more subjective compared to image retrieval tasks and the set of known similarities is very scarce. To tackle this challenge, in this paper, we propose a simple but effective segment-based augmentation strategy to introduce artificially similar logos for training deep networks for logo retrieval. In this novel augmentation strategy, we first find segments in a logo and apply transformations such as rotation, scaling, and color change, on the segments, unlike the conventional image-level augmentation strategies. Moreover, we evaluate whether the recently introduced ranking-based loss function, Smooth-AP, is a better approach for learning similarity for logo retrieval. On the large scale METU Trademark Dataset, we show that (i) our segment-based augmentation strategy improves retrieval performance compared to the baseline model or image-level augmentation strategies, and (ii) Smooth-AP indeed performs better than conventional losses for logo retrieval. 
<p align="center">
<img src=https://github.com/blisgard/SegmentBasedAugmentation/blob/main/figures/teaser_figures.png  width="450" height="300">
  
  <p align="center">Figure: A comparison of the proposed segment-level augmentation method with conventional image-level augmentation.</p>
</p>

## Overview of the Method

We perform segment level augmentation by following these steps (see the following figure for an illustration):
1. **Logo segmentation**: Segment a logo into its regions. 
2. **Segment selection**: Select a random segment.
3. **Segment transformation**: Apply transformation (rotation, color change, removal) on the selected segment.

<p align="center" >
  <img src=https://github.com/blisgard/SegmentBasedAugmentation/blob/main/figures/sample_augmentations.png  width="300" height="360">
  
  <p align="center">Figure: Sample augmentations for a logo.</p>
</p>

## Presentation Video

You can access a short presentation [here](https://www.youtube.com/watch?v=06d4OvMqmWg). 

## Sample Experimental Results

**Table: A comparison between the effects of Image-Level (Horizontal Flip, Vertical Flip) and Segment-Level
Augmentation.**:

| Method        | Normalized Average Rank (lower better)     | 
| ------------- |:-------------:| 
| Baseline(No augmentation)    |  0.102 | 
| Triplet Loss(No augmentation)      | 0.053      | 
| Triplet Loss(Image-level aug.)|  0.051      |  
| Triplet Loss(S. Color, S. Removal)|  0.046     |  
| Smooth-AP Loss(No augmentation)|  0.046      |  
| Smooth-AP Loss(Image-level aug.)|  0.044      |  
| Smooth-AP Loss(S. Color) |  0.040      |  

## Sample Visual Results
![](https://github.com/blisgard/SegmentBasedAugmentation/blob/main/figures/visual_results_1.png)
![](https://github.com/blisgard/SegmentBasedAugmentation/blob/main/figures/visual_results_2.png)

## Dataset
In this repository, [METU Trademark Dataset](https://github.com/neouyghur/METU-TRADEMARK-DATASET) is used. The dataset is available per request only. If you are a researcher at a university (or a graduate student) interested in the dataset for research purposes, please contact Sinan Kalkan with your intention.

## Running the Code

### Dependencies

Dependencies can be found in requirements.txt. 

Use `pip3 install requirements.txt`


### Training the model

python3 Standard_Training.py  --bs 512 --lr 0.001 --n_epochs 50

## Recommended Citation

Feyza Yavuz and Sinan Kalkan, "Segment Augmentation and Differentiable Ranking for Logo Retrieval", 26th International Conference on Pattern Recognition (ICPR), 2022.

```
@inproceedings{YavuzKalkan,
  title={Segment Augmentation and Differentiable Ranking for Logo Retrieval},
  author={Yavuz, Feyza and Kalkan, Sinan},
  booktitle={26th International Conference on Pattern Recognition (ICPR)},
  year={2022},
}
```

## Contact

Feyza Yavuz, E-mail: <feyza.yavuz@metu.edu.tr>

