# Mind Peak Challenge
This repository contains the code and classification results when applying CNNs to the Mind Peak Challenge dataset. Experiments include training a small CNN from scratch and performing transfer learning, where features of the [VGG11](https://pytorch.org/docs/stable/torchvision/models.html) net are used.

Because of the small size of the dataset (400 samples that are split into 80% for training and 20% for testing purposes), models that are trained from scratch tend to overfit on the training data and do not generalise well. Transfer learning, where pre-trained feature maps are used and only the classifier is trained, mitigates this issue and leads to better generalisation on the test set as well as a more stable training procedure. To further improve generalisation, data-augmentation is used when training both CNN architectures.

The models are implemented in [PyTorch](https://github.com/pytorch/pytorch).

## Running the Code

### Preprocessing

Because the samples themselves are very large in size, preprocessing involves resizing them to a lower dimensionality in order to reduce computation cost.

To preprocess the images run:
```
python preprocessing.py --source-root={DIR_TO_ORIGINAL_DATASET} --dest-root={DESTINATION_DIR} --width-height={MAX_WIDTH_HEIGHT}
```
The experiments in this repository use samples downscaled to --width-height=128, which are included in "preprocessed/".

### Training / Testing the Models

Models can be trained by running the main.py script. Options include training a simple CNN trained from scratch and a transfer learning CNN (using VGG11).

The simple CNN can be trained by:
```
python main.py --model-type=CNN --dataset_dir="preprocessed/" --model-name=SimpleCNN --mode="train"
```

The transfer learning CNN model can be trained by:
```
python main.py --model-type=TransferCNN --dataset_dir="preprocessed/" --model-name=TransferCNN --mode="train"
```

For additional options (such as setting hyperparameters) see:
```
python main.py --h
```

To test the model and calculate the accuracy after training, run the main script with the argument:
```
--mode="test"
```

### Results

Trained models and tensorboard results can be found in "experiments/". While the simple CNN does not converge well (test loss fluctuating heavily when trained for more than 70 epochs due to overfitting) and reaches a test accurracy between 50% - 60%, the transfer learning CNN shows stable convergence and achieves a test accuracy between 72% - 77%.

To look at the results in tensorboard run:
```
tensorboard --logdir=experiments
```
