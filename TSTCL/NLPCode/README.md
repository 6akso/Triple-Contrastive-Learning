# Triple Contrastive Learning Representation Boosting for Supervised Multiclass Tasks

## Introduction
Supervised contrastive learning has been widely demonstrated to be effective for extracting sample representations, thereby improving the performance of downstream tasks. However, while supervised contrastive learning can fully exploit supervised signals in datasets containing only up to two classes, it suffers from under-utilization of supervised signals in datasets containing multiple classes (number of classes greater than two). This under-utilization results in a reduction in the inter-class distance, which affects the generalization effect.

In this study, we propose a triple-supervised contrastive learning concept and apply it to existing supervised contrastive learning methods for natural language processing and computer vision tasks. Our constraint requires that the distance between negative samples with different labels in the set of negative samples generated from an identical anchor point be pulled apart, thus expanding the existing binary constraints on the loss function of supervised contrastive learning (anchor-positive, anchor-negative) to a triple constraint (anchor-positive, anchor-negative, between-negative). This can help the model learn a more realistic distribution in the sample space, thus improving its generalization ability.

We conducted experiments on eight publicly available multi-class datasets and verified that our triple constraints can effectively improve the model’s ability to extract sample representations by measuring the classification accuracy and MarcoF1 score, analyzing the inter- and intra-class distances, and visualizing the clusters formed by the representations.

## Installation
1. Clone this repository
   ```bash
   git https://github.com/6akso/Xianshuai-Li.git


## Usage
To run the baseline model:
python BaseLine.py

To run the TST-CL model:
python TST-CL.py

To run the TST-DualCL model:
python TST-DualCL.py

Email: 1849667739@qq.com
