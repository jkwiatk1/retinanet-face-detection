# RetinaNet for Face Detection
* white paper: [link](https://arxiv.org/pdf/1708.02002v2.pdf)
* article: [link](https://towardsdatascience.com/review-retinanet-focal-loss-object-detection-38fba6afabe4)

## ResNet50
Backbone created using the PyTorch model with pre-trained weights. 

## Feature Pyramid Network
Created based on:
* white paper: [link](https://openaccess.thecvf.com/content_cvpr_2017/papers/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.pdf)
* article_1: [link](https://jonathan-hui.medium.com/understanding-feature-pyramid-networks-for-object-detection-fpn-45b227b9106c)
* article_2: [link](https://towardsdatascience.com/review-fpn-feature-pyramid-network-object-detection-262fc7482610)
* article_3: [link](https://medium.com/@freshtechyy/fusing-backbone-features-using-feature-pyramid-network-fpn-c652aa6a264b)

FPN extracts feature maps and later feeds into a detector, like RPN.

## Region Proposal Network 
RPN applies a sliding window over the feature maps to make predictions on the objectness (has an object or not) and the object boundary box at each location.

In the FPN framework, for each scale level (say P4), a 3 × 3 convolution filter is applied over the feature maps followed by separate 1 × 1 convolution for objectness predictions and boundary box regression. These 3 × 3 and 1 × 1 convolutional layers are called the RPN head. The same head is applied to all different scale levels of feature maps.