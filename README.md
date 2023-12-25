# GVAlign
Robust Feature Learning and Global Variance-Driven Classifier Alignment for Long-Tail Class Incremental Learning

### Dependencies
All library details given in long_tail_cil.yml file. To install conda environment run the following command.
```
$conda env create -f long_tail_cil.yml
```

### Training
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



