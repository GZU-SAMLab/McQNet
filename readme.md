# Meta-contrastive Learning with Support-based Query Interaction for Few-shot Fine-grained Visual Classification

> Few-shot fine-grained visual classification combines the challenge of scarcity of training data for few-shot tasks and intrinsic property with inter-intra variance for fine-grained tasks. To alleviate this problem, we first propose a support-based query interaction module and a meta-contrastive method. Extensive experiments demonstrate that our method performs well on both common fine-grained datasets and few-shot datasets in different experimental settings.

# Our Major Contributions

> 1. We propose a novel meta-contrastive framework for finegrained few-shot learning, further improving the performance of few-shot learning and fine-grained tasks. This may be the first to apply contrastive learning to finegrained few-shot tasks.
> 2. We develop a support-based query interaction method that not only establishes the dependencies between support samples and query samples, but also enhances query features by leveraging support features.
> 3. We conduct comprehensive experiments on three benchmark fine-grained datasets and a common few-shot benchmark dataset. Our experimental results achieve a SOTA performance on these datasets.

# Architecture of our proposed Network

![image-20231016210225341](./pictures/1.png)

> The network is composed of three main parts, including the feature extractor module, the support-based query interaction module (SQIM), and the meta-contrastive module (MCM). Among them, SQIM and MCM are the core of the proposed method. Indeed, SQIM mines latent knowledge by interacting with query features and support features. MCM enhances the model's learning ability to perform fine-grained few-shot tasks by incorporating the concept of contrastive learning.

# Support-based Query Interaction Module

<img src="./pictures/3.png" alt="image-20231016212731788" style="zoom:150%;" />

> The detailed flow of support-based query interaction module.

# Meta-Contrastive Module

<img src="./pictures/4.png" alt="image-20231016212851060" style="zoom:150%;" />

> The detailed flow of meta-contrastive module.

# Environment Requirements

> To run the project, you need the following environment configuration

``````
numpy==1.21.5
pandas==1.2.4
resnet==0.1
torch==1.10.2
torchvision==0.11.3
tqdm==4.66.1
``````
# Dataset
>download the dataset from [http://www.vision.caltech.edu/visipedia/CUB-200-2011.html](http://www.vision.caltech.edu/datasets/cub_200_2011/)
>Caltech-UCSD Birds-200-2011 (CUB-200-2011) is an extended version of the CUB-200 dataset, with roughly double the number of images per class and new part location annotations.

Number of categories: 200
Number of images: 11,788
Annotations per image: 15 Part Locations, 312 Binary Attributes, 1 Bounding Box

# Result

![image-20231016210844054](./pictures/2.png)

> 5-way few-shot classification performance (%) on the CUB, NAB, and Stanford dogs datasets. The ± denotes that the results are reported with 95% confidence intervals over 1000 test episodes. The highest average accuracy of each column is marked in bold.

