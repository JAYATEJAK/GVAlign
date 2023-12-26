# GVAlign
This is the official repo for the paper:

***Robust Feature Learning and Global Variance-Driven Classifier Alignment for Long-Tail Class Incremental Learning*** (WACV 2024) [Paper link](https://openaccess.thecvf.com/content/WACV2024/papers/Kalla_Robust_Feature_Learning_and_Global_Variance-Driven_Classifier_Alignment_for_Long-Tail_WACV_2024_paper.pdf)




> **Abstract** This paper introduces a two-stage framework designed to enhance long-tail class incremental learning, enabling the model to progressively learn new classes, while mitigating catastrophic forgetting in the context of long-tailed data distributions. Addressing the challenge posed by the underrepresentation of tail classes in long-tail class incremental learning, our approach achieves classifier alignment by leveraging global variance as an informative measure and class prototypes in the second stage. This process effectively captures class properties and eliminates the need for data balancing or additional layer tuning. Alongside traditional class incremental learning losses in the first stage, the proposed approach incorporates mixup classes to learn robust feature representations, ensuring smoother boundaries. The proposed framework can seamlessly integrate as a module with any class incremental learning method to effectively handle long-tail class incremental learning scenarios. Extensive experimentation on the CIFAR-100 and ImageNet-Subset datasets validates the approach’s efficacy, showcasing its superiority over state-of-the-art techniques across various long-tail CIL settings.

<p align="center">
  <img src="https://github.com/JAYATEJAK/GVAlign/blob/main/Figures/WACV_2stage-Page-6-Page-5.drawio.svg" alt="Sublime's custom image"/>
</p> 

### Dependencies
All library details given in long_tail_cil.yml file. To install conda environment run the following command.
```
$conda env create -f long_tail_cil.yml
```

### Training
```
$bash ./scripts/script_cifar100_no_gridsearch.sh <approach> <GPU_ID> <Dataset> <Distribution type> <# Base classes> <# tasks>
```
#### CIFAR-100 - shuffled long tail
```
$bash ./scripts/script_cifar100_no_gridsearch.sh lucir_gvalign_2stage 0 cifar100 lt 50 6
```

#### CIFAR-100 - ordered long tail
```
$bash ./scripts/script_cifar100_no_gridsearch.sh lucir_gvalign_2stage 0 cifar100 ltio 50 6
```

#### CIFAR-100 -  conventional
```
$bash ./scripts/script_cifar100_no_gridsearch.sh lucir_gvalign_2stage 0 cifar100 conv 50 6
```
#### ImageNet100 - shuffled long tail
```
$bash ./scripts/script_cifar100_no_gridsearch.sh lucir_gvalign_2stage 0 imagenet_subset lt 50 6
```

#### ImageNet100 - ordered long tail
```
$bash ./scripts/script_cifar100_no_gridsearch.sh lucir_gvalign_2stage 0 imagenet_subset ltio 50 6
```

#### ImageNet100 -  conventional
```
$bash ./scripts/script_cifar100_no_gridsearch.sh lucir_gvalign_2stage 0 imagenet_subset conv 50 6
```
### Acknowledgement
This repository is built on the Long-Tail CIL Repo (https://github.com/xialeiliu/Long-Tailed-CIL). Many thanks to the authors for releasing the code.



