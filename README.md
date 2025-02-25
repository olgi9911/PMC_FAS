# Progressive Modality Cooperation for Multi-Modality Domain Adaptation
This is an implementation of the paper [Progressive Modality Cooperation for Multi-Modality Domain Adaptation](https://ieeexplore.ieee.org/abstract/document/9334409).

## Usage
- `train.py`: Performs initial training using the domain adversarial learning strategy described in [DANN](https://arxiv.org/abs/1505.07818).
- `train_mmg.py`: Trains the MMG module using labeled source samples with all modalities and unlabeled target samples with one or two missing modalities.
- `adapt.py`: Performs PMC adaptation.
- `test.py`: Conducts inference at each **model_save_epoch** or **model_save_step**.
- `dataloader.py`: Dataloader for training/testing.
- `utils.py`: Contains helper functions.
- `models/dann.py`: Defines the DANN (Domain-Adversarial Training of Neural Networks) model.
- `models/mmg.py`: Implements the Multi-Modality Generation (MMG) module, which consists of a UNet with a ResNet50 pretrained encoder and a domain classifier.

One can run the following command to train the model:
```console
python train.py --train_dataset [DATASET1/DATASET2/...] --test_dataset [DATASET1/DATASET2/...] --epochs [EPOCH] --batch_size [BATCH_SIZE] --lr [LEARNING_RATE] --trade_off [TRADE_OFF]
```

To train the MMG module:
```console
python train_mmg.py --train_dataset [DATASET1/DATASET2/...] --test_dataset [DATASET1/DATASET2/...] --missing [depth/ir/depth+ir] --epochs [EPOCH] --batch_size [BATCH_SIZE]
```

To perform PMC adaptation:
```console
python adapt.py --train_dataset [DATASET1/DATASET2/...] --test_dataset [DATASET1/DATASET2/...] --epochs [EPOCH] --batch_size [BATCH_SIZE] --lr [LEARNING_RATE] --trade_off [TRADE_OFF]
```

To test the model and log the results:
```console
python test.py --train_dataset [DATASET1/DATASET2/...] --test_dataset [DATASET1/DATASET2/...] --missing [DATASET/none] --pmc [BOOLEAN]
```

Note that one needs to set `pmc` flag to *True* after PCM adaptation to correctly load the model weights.