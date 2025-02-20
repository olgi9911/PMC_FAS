# Progressive Modality Cooperation for Multi-Modality Domain Adaptation
This is an implementation of the paper [Progressive Modality Cooperation for Multi-Modality Domain Adaptation](https://ieeexplore.ieee.org/abstract/document/9334409).

## Usage
- `train.py`: Initial training by following the domain adversarial learning strategy as decsribed in [DANN](https://arxiv.org/abs/1505.07818).
- `adapt.py`: Perform PMC adaptation.
- `test.py`: Perform inference per **model_save_epoch** or **model_save_step**.
- `dataloader.py`: Dataloader for training/testing.
- `models/dann.py`: DANN (Domain-Adversarial Training of Neural Networks) is defined here.

One can run the following command to train the model:
```console
python train.py --train_dataset [DATASET1/DATASET2/...] --test_dataset [DATASET1/DATASET2/...] --epochs [EPOCH] --batch_size [BATCH_SIZE] --lr [LEARNING_RATE] --trade_off [TRADE_OFF]
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