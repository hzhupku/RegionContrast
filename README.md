# Region-aware Contrastive Learning for Semantic Segmentation, ICCV 2021

## Abstract
Recent works have made great success in semantic segmentation by exploiting contextual information in a local or global manner within individual image and supervising the model with pixel-wise cross entropy loss. However, from the holistic view of the whole dataset, semantic relations not only exist inside one single image, but also prevail in the whole training data, which makes solely considering intra-image correlations insufficient. Inspired by recent progress in unsupervised contrastive learning, we propose the region-aware contrastive learning (RegionContrast) for semantic segmentation in the supervised manner. In order to enhance the similarity of semantically similar pixels while keeping the discrimination from others, we employ contrastive learning to realize this objective. With the help of memory bank, we explore to store all the representative features into the memory. Without loss of generality, to efficiently incorporate all training data into the memory bank while avoiding taking too much computation resource, we propose to construct region centers to represent features from different categories for every image. Hence, the proposed region-aware contrastive learning is performed in a region level for all the training data, which saves much more memory than methods exploring the pixel-level relations. The proposed RegionContrast brings little computation cost during training and requires no extra overhead for testing. Extensive experiments demonstrate that our method achieves state-of-the-art performance on three benchmark datasets including Cityscapes, ADE20K and COCO Stuff. For more details, please refer to our ICCV paper ([paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Hu_Region-Aware_Contrastive_Learning_for_Semantic_Segmentation_ICCV_2021_paper.pdf)).

![image](https://github.com/hzhupku/RegionContrast/blob/main/arch.png)

## Installation
Check [INSTALL.md](INSTALL.md) for installation instructions.

## Training and Evaluation
```bash
cd experiments/v3_contrast
bash train.sh
```
## Citation
```
@InProceedings{Hu_2021_ICCV,
    author    = {Hu, Hanzhe and Cui, Jinshi and Wang, Liwei},
    title     = {Region-Aware Contrastive Learning for Semantic Segmentation},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {16291-16301}
}
```
### TODO
- [ ] Dynamic Sampling