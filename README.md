# AAN

Class-level Anchor Alignment (CAA) on top of AlignedNorm for CLIP few-shot / base-to-novel classification.

This repository keeps only the files needed to train and evaluate ALIGNEDNORM with class-level anchors.

## Core files

- train.py: training / eval entry
- trainers/alignednorm.py: ALIGNEDNORM trainer + CAA losses
- trainers/alignednorm_config.py: dataset-specific hyper-parameters
- clip/model_alignednorm.py, clip/prs_hook.py: CLIP backbone and PRS hooks
- configs/trainers/ALIGNEDNORM/: method configs
- scripts/alignednorm/: base-to-novel / few-shot / cross-dataset scripts

## Base-to-novel (16-shot)

Train on base classes:

python train.py --root DATASET_ROOT --seed 1 --trainer ALIGNEDNORM --dataset-config-file configs/datasets/stanford_cars.yaml --config-file configs/trainers/ALIGNEDNORM/vit_b16.yaml --output-dir output/base2new/train_base/stanford_cars/shots_16/ALIGNEDNORM/vit_b16_caa/seed1 DATASET.NUM_SHOTS 16 DATASET.SUBSAMPLE_CLASSES base TASK B2N

Evaluate on novel classes:

python train.py --root DATASET_ROOT --seed 1 --trainer ALIGNEDNORM --dataset-config-file configs/datasets/stanford_cars.yaml --config-file configs/trainers/ALIGNEDNORM/vit_b16.yaml --output-dir output/base2new/test_new/stanford_cars/shots_16/ALIGNEDNORM/vit_b16_caa/seed1 --model-dir output/base2new/train_base/stanford_cars/shots_16/ALIGNEDNORM/vit_b16_caa/seed1 --eval-only DATASET.NUM_SHOTS 16 DATASET.SUBSAMPLE_CLASSES new TASK B2N

## Notes

- Depends on Dassl.pytorch.
- Checkpoints, logs, and datasets are not included.
