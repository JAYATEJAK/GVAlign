#!/bin/bash
##  ./scripts/script_cifar100_no_gridsearch.sh lucir 0 cifar100 conv 50 6
##  ./scripts/script_cifar100_no_gridsearch.sh lucir_2stage 0 cifar100 conv 50 6

##  ./scripts/script_cifar100_no_gridsearch.sh lucir 0 cifar100 conv 50 11
##  ./scripts/script_cifar100_no_gridsearch.sh lucir_2stage 0 cifar100 conv 50 11

##  ./scripts/sconv
##  ./scripts/script_cifar100_no_gridsearch.sh lucir_2stage 0 imagenet_subset conv 50 6

##  ./scripts/script_cifar100_no_gridsearch.sh lucir 0 imagenet_subset conv 50 11
##  ./scripts/script_cifar100_no_gridsearch.sh lucir_2stage 0 imagenet_subset conv 50 11

### bash ./scripts/script_cifar100_no_gridsearch.sh lucir_mixup_self_supervision_continous_proto_aug1 2 cifar100 lt 50 6

################## conv lt ltio

if [ "$1" != "" ]; then
    echo "Running approach: $1"
else
    echo "No approach has been assigned."
fi
if [ "$2" != "" ]; then
    echo "Running on gpu: $2"
else
    echo "No gpu has been assigned."
fi

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && cd .. && pwd )"
SRC_DIR="$PROJECT_DIR/src"
echo "Project dir: $PROJECT_DIR"
echo "Sources dir: $SRC_DIR"
# RESULTS_DIR="results/stochastic_classifier_single_variance_for_all_classes_with_scale_ucir_300_epochs_100_200_250"
# RESULTS_DIR="results/ldam_stochastic_basic_ucir_300_epochs_100_200_250"
RESULTS_DIR="results/self_sup_expts_two_rotations_300epochs"
RESULTS_DIR="results/test_loss_expts_on_self_supervision_500epochs_lamda_1_for_ss_all_tasks_mixup_only_base_dist_only_normal_images_cont_mixup"
RESULTS_DIR="results/checking_with_mixup_stochastic"
RESULTS_DIR="results/mixup_with_proto_augmentation_last_step_var_proto_update"
RESULTS_DIR="results/mixup_with_all_tasks_protoaug_500epochs_before_test_100_epochs_tune_base_sigma"
RESULTS_DIR="results/adapt_before_test_expts"
RESULTS_DIR="results/adapt_before_test_expts_continous_mix_up_analyzing_step1_300epochs"
RESULTS_DIR="results/adapt_before_test_expts_continous_mix_up_analyzing_confusion_matrix"
RESULTS_DIR="results/adapt_before_test_expts_continous_mix_up_500epochs_50_5_5_5_with_tune_rerun"
RESULTS_DIR="results/mixup_intermediate_nodes"
RESULTS_DIR="results/test_podnet"
RESULTS_DIR="results/mixup_stage1_proto_aug_stage2_with_herding"
RESULTS_DIR="results/ablation_with_exemplars_1"
RESULTS_DIR="results/ablation_2stage"
RESULTS_DIR="results/tsne_plots"
RESULTS_DIR="results/ucir_lws_mixup"
RESULTS_DIR="results/wacv_gv_align"
if [ "$7" != "" ]; then
    RESULTS_DIR=$7
else
    echo "No results dir is given. Default will be used."
fi
echo "Results dir: $RESULTS_DIR"
# mkdir $RESULTS_DIR
for SEED in 0 
do
  if [ "$3" = "base" ]; then
          PYTHONPATH=$SRC_DIR python3 -u $SRC_DIR/main_incremental.py --exp-name no_gs_base_${SEED} \
                 --datasets cifar100_icarl --num-tasks 10 --network resnet32 --seed $SEED \
                 --nepochs 500 --batch-size 128 --results-path $RESULTS_DIR \
                 --approach $1 --gpu $2 --lr 0.1 --lr-min 1e-5 --lr-factor 3 --momentum 0.9 \
                 --weight-decay 0.0002 --lr-patience 15
    elif [ "$3" = "cifar100" ]; then
        PYTHONPATH=$SRC_DIR python3 -u $SRC_DIR/main_incremental.py --exp-name no_gs_fixd_${SEED} \
                --datasets "$3_$4" --num-tasks $6 --network resnet32 --seed $SEED \
                --nepochs 500 --batch-size 128 --results-path "$PROJECT_DIR/cifar100/$5base_$6tasks/$RESULTS_DIR" \
                --approach $1 --gpu $2 --lr 0.1 --lr-factor 10 --momentum 0.9 \
                --weight-decay 5e-4 \
                --nc-first-task $5 \
                --num-exemplars-per-class 20 --exemplar-selection herding
    elif [ "$3" = "imagenet_subset" ]; then
        PYTHONPATH=$SRC_DIR python3 -u $SRC_DIR/main_incremental.py --exp-name no_gs_fixd_${SEED} \
                --datasets "$3_$4" --num-tasks $6 --network resnet18 --seed $SEED \
                --nepochs 90 --batch-size 64 --results-path "$PROJECT_DIR/imagenet_subset/$5base_$6tasks/$RESULTS_DIR" \
                --approach $1 --gpu $2 --lr 0.1 --lr-factor 10 --momentum 0.9 \
                --weight-decay 1e-4 \
                --nc-first-task $5 --schedule_step 30 60 \
                --num-exemplars-per-class 20 --exemplar-selection herding
  else
          echo "No scenario provided."
  fi
done