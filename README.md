# Deep Learning as a Mixed Convex-Combinatorial Optimization Problem

This repository contains the code for running the experiments in the paper "Deep Learning as a Mixed Convex-Combinatorial Optimization Problem" by Friesen & Domingos, ICLR 2018. [(ICLR 2018 link)](https://openreview.net/forum?id=B1Lc-Gb0Z) [(arxiv link)](https://arxiv.org/abs/1710.11573)

Specifically, it contains my implementation of mini-batch feasible target propagation (ftprop-mb) for training multiple different convnets with either step or staircase-like activation functions. Models can be learned using either a soft hinge loss or the saturating straight-through estimator (SSTE). See paper for details.

## Dependencies

I ran this project in a conda virtual environment on Ubuntu 16.04 with CUDA 8.0. I've also (very briefly) tested it on Mac OS X. This project requires the following dependencies:
* [pytorch with torchvision](http://pytorch.org/). I built this from source from the current master at the time (pytorch revision 76a282d22880de9f37f17af31e40e18994fa9870, torchvision revision 88e81cea5795d80ed63bc68675b1989691801f57). Note that you need at least a fairly recent version of PyTorch (e.g., 0.3), which conda install does not provide.
* [tensorflow with tensorboard](https://www.tensorflow.org/install/) (This is not critical, however, and you can comment out all references to TF and TB if you don't want to install it)
* setproctitle (e.g., [via conda](https://anaconda.org/conda-forge/setproctitle))
* [scipy](https://www.scipy.org/)

## Running the experiments

There are a number of command-line arguments available, including 
* --tp-rule <TPRule_name>, specify the targetprop rule to use to train (e.g., SoftHinge, SSTE)
* --nonlin <nonlinearity_name>, specify the nonlinearity to use in the network (e.g., step11, staircase3, relu, threshrelu)
* --gpus <list_of_GPU_indices_separated_by_spaces>, specify the GPU(s) that this run should use for training + testing (supports data parallelization across multiple GPUs)
* --nworkers <#workers>, the number of workers (i.e., threads) to use for loading data from disk
* --data-root <path/to/imagenet>, for ImageNet only, specify the path to the ImageNet dataset
* --no-cuda, specify this flag to run on CPU only
* --resume <path/to/checkpoint>, resume training from a saved checkpoint
* --no-save, specify this flag to disable any logging or checkpointing to disk
* --download, specify this flag to download MNIST or CIFAR if it's not found
* --no-val, specify this to NOT create a validation set and use that for testing (only use this for the final run!)
* --plot-test, specify this to plot the test accuracy each iteration (only use this for the final run!)

Please see the code or run the file for a full list.


To reproduce the experiments from the paper, use the following commands:

### 4-layer convnet, CIFAR-10

```
# Step activation with FTP-SH
python3 train.py --ds=cifar10 --arch=convnet4 --wtdecay=5e-4 --lr-decay=0.1 --epochs 300 --lr-decay-epochs 200 250 --momentum=0.9 --opt=adam --lr=0.00025 --loss=crossent --nonlin=step11 --tp-rule=SoftHinge --nworkers=6 --plot-test --test-final-model --no-val --download

# Step activation with SSTE
python3 train.py --ds=cifar10 --arch=convnet4 --wtdecay=5e-4 --lr-decay=0.1 --epochs 300 --lr-decay-epochs 200 250 --momentum=0.9 --opt=adam --lr=0.00025 --loss=crossent --nonlin=step11 --tp-rule=SSTE --nworkers=6 --plot-test --test-final-model --no-val --download

# 2-bit quantized ReLU (staircase3) activation with FTP-SH
python3 train.py --ds=cifar10 --arch=convnet4 --wtdecay=5e-4 --lr-decay=0.1 --epochs 300 --lr-decay-epochs 200 250 --momentum=0.9 --opt=adam --lr=0.00025 --loss=crossent --nonlin=staircase3 --tp-rule=SoftHinge --nworkers=6 --plot-test --test-final-model --no-val --download

# 2-bit quantized ReLU (staircase3) activation with SSTE
python3 train.py --ds=cifar10 --arch=convnet4 --wtdecay=5e-4 --lr-decay=0.1 --epochs 300 --lr-decay-epochs 200 250 --momentum=0.9 --opt=adam --lr=0.00025 --loss=crossent --nonlin=staircase3 --tp-rule=SSTE --nworkers=6 --plot-test --test-final-model --no-val --download

# ReLU activation
python3 train.py --ds=cifar10 --arch=convnet4 --wtdecay=5e-4 --lr-decay=0.1 --epochs 300 --lr-decay-epochs 200 250 --momentum=0.9 --opt=adam --lr=0.00025 --loss=crossent --nonlin=relu --nworkers=6 --plot-test --test-final-model --no-val --download

# Thresholded-ReLU activation
python3 train.py --ds=cifar10 --arch=convnet4 --wtdecay=5e-4 --lr-decay=0.1 --epochs 300 --lr-decay-epochs 200 250 --momentum=0.9 --opt=adam --lr=0.00025 --loss=crossent --nonlin=threshrelu --nworkers=6 --plot-test --test-final-model --no-val --download
```

### 8-layer convnet, CIFAR-10

```
# Step activation with FTP-SH
python3 train.py --ds=cifar10 --arch=convnet8 --wtdecay=0.0005 --lr-decay=0.1 --epochs 300 --lr-decay-epochs 200 250 --momentum=0.9 --opt=adam --lr=1e-3 --loss=crossent --nonlin=staircase3 --tp-rule=SoftHinge --use-bn --nworkers=6 --plot-test --no-val --test-final-model --download

# Step activation with SSTE
python3 train.py --ds=cifar10 --arch=convnet8 --wtdecay=0.0005 --lr-decay=0.1 --epochs 300 --lr-decay-epochs 200 250 --momentum=0.9 --opt=adam --lr=1e-3 --loss=crossent --nonlin=staircase3 --tp-rule=SSTE --use-bn --nworkers=6 --plot-test --no-val --test-final-model --download

# 2-bit quantized ReLU (staircase3) activation with FTP-SH
python3 train.py --ds=cifar10 --arch=convnet8 --wtdecay=0.0005 --lr-decay=0.1 --epochs 300 --lr-decay-epochs 200 250 --momentum=0.9 --opt=adam --lr=1e-3 --loss=crossent --nonlin=staircase3 --tp-rule=SoftHinge --use-bn --nworkers=6 --plot-test --no-val --test-final-model --download

# 2-bit quantized ReLU (staircase3) activation with SSTE
python3 train.py --ds=cifar10 --arch=convnet8 --wtdecay=0.0005 --lr-decay=0.1 --epochs 300 --lr-decay-epochs 200 250 --momentum=0.9 --opt=adam --lr=1e-3 --loss=crossent --nonlin=staircase3 --tp-rule=SSTE --use-bn --nworkers=6 --plot-test --no-val --test-final-model --download

# ReLU activation
python3 train.py --ds=cifar10 --arch=convnet8 --wtdecay=0.0005 --lr-decay=0.1 --epochs 300 --lr-decay-epochs 200 250 --momentum=0.9 --opt=adam --lr=1e-3 --loss=crossent --nonlin=relu --use-bn --nworkers=6 --plot-test --no-val --test-final-model --download

# Thresholded-ReLU activation
train.py --ds=cifar10 --arch=convnet8 --wtdecay=0.0005 --lr-decay=0.1 --epochs 300 --lr-decay-epochs 200 250 --momentum=0.9 --opt=adam --lr=1e-3 --loss=crossent --nonlin=threshrelu --use-bn --nworkers=6 --plot-test --no-val --test-final-model --download
```


### AlexNet (from DoReFa paper), ImageNet

Note: you must already have ImageNet downloaded to your computer to run these experiments. Also, I preprocessed the ImageNet images to scale them to 256x256 but this could also be done with a transformation when loading the dataset (you'll need to add this however).

```
# Step activation with FTP-SH
python3 train.py --ds=imagenet --arch=alexnet_drf --batch=128 --wtdecay=5e-6 --lr-decay=0.1 --epochs 80 --lr-decay-epochs 56 64 --momentum=0.9 --opt=adam --lr=1e-4 --loss=crossent --nonlin=step11 --tp-rule=SoftHinge --nworkers=12 --use-bn --no-val --plot-test --data-root=<path/to/imagenet>

# Step activation with SSTE
python3 train.py --ds=imagenet --arch=alexnet_drf --batch=128 --wtdecay=5e-6 --lr-decay=0.1 --epochs 80 --lr-decay-epochs 56 64 --momentum=0.9 --opt=adam --lr=1e-4 --loss=crossent --nonlin=step11 --tp-rule=SSTE --nworkers=12 --use-bn --no-val --plot-test --data-root=<path/to/imagenet>

# 2-bit quantized ReLU (staircase3) activation with FTP-SH
python3 train.py --ds=imagenet --arch=alexnet_drf --batch=128 --wtdecay=5e-5 --lr-decay=0.1 --epochs 80 --lr-decay-epochs 56 64 --momentum=0.9 --opt=adam --lr=1e-4 --loss=crossent --nonlin=staircase3 --tp-rule=SoftHinge --nworkers=12 --use-bn --no-val --plot-test --data-root=<path/to/imagenet>

# 2-bit quantized ReLU (staircase3) activation with SSTE
python3 train.py --ds=imagenet --arch=alexnet_drf --batch=128 --wtdecay=5e-5 --lr-decay=0.1 --epochs 80 --lr-decay-epochs 56 64 --momentum=0.9 --opt=adam --lr=1e-4 --loss=crossent --nonlin=staircase3 --tp-rule=SSTE --nworkers=12 --use-bn --no-val --plot-test --data-root=<path/to/imagenet>

# ReLU activation
python3 train.py --ds=imagenet --arch=alexnet_drf --batch=128 --wtdecay=5e-4 --lr-decay=0.1 --epochs 80 --lr-decay-epochs 56 64 --momentum=0.9 --opt=adam --lr=1e-4 --loss=crossent --nonlin=relu --nworkers=12 --use-bn --no-val --plot-test --data-root=<path/to/imagenet>

# Thresholded-ReLU activation
python3 train.py --ds=imagenet --arch=alexnet_drf --batch=128 --wtdecay=5e-4 --lr-decay=0.1 --epochs 80 --lr-decay-epochs 56 64 --momentum=0.9 --opt=adam --lr=1e-4 --loss=crossent --nonlin=threshrelu --nworkers=12 --use-bn --no-val --plot-test --data-root=<path/to/imagenet>
```

### ResNet-18, ImageNet

Note: you must already have ImageNet downloaded to your computer to run these experiments. Also, I preprocessed the ImageNet images to scale them to 256x256 but this could also be done with a transformation when loading the dataset (you'll need to add this however).

```
# Step activation with FTP-SH
python3 train.py --ds=imagenet --arch=resnet18 --batch=256 --wtdecay=5e-7 --lr-decay=0.1 --epochs 90 --lr-decay-epochs 30 60 --momentum=0.9 --opt=sgd --lr=0.1 --loss=crossent --nonlin=step11 --tp-rule=SoftHinge --nworkers=12 --use-bn --no-val --plot-test --data-root=<path/to/imagenet>

# Step activation with SSTE
python3 train.py --ds=imagenet --arch=resnet18 --batch=256 --wtdecay=5e-7 --lr-decay=0.1 --epochs 90 --lr-decay-epochs 30 60 --momentum=0.9 --opt=sgd --lr=0.1 --loss=crossent --nonlin=step11 --tp-rule=SSTE --nworkers=12 --use-bn --no-val --plot-test --data-root=<path/to/imagenet>

# 3-bit quantized ReLU (staircase3) activation with FTP-SH
python3 train.py --ds=imagenet --arch=resnet18 --batch=256 --wtdecay=1e-5 --lr-decay=0.1 --epochs 90 --lr-decay-epochs 30 60 --momentum=0.9 --opt=sgd --lr=0.1 --loss=crossent --nonlin=staircase --tp-rule=SoftHinge --nworkers=12 --use-bn --no-val --plot-test --data-root=<path/to/imagenet>

# 3-bit quantized ReLU (staircase3) activation with SSTE
python3 train.py --ds=imagenet --arch=resnet18 --batch=256 --wtdecay=1e-5 --lr-decay=0.1 --epochs 90 --lr-decay-epochs 30 60 --momentum=0.9 --opt=sgd --lr=0.1 --loss=crossent --nonlin=staircase --tp-rule=SSTE --nworkers=12 --use-bn --no-val --plot-test --data-root=<path/to/imagenet>

# ReLU activation
python3 train.py --ds=imagenet --arch=resnet18 --batch=256 --wtdecay=1e-4 --lr-decay=0.1 --epochs 90 --lr-decay-epochs 30 60 --momentum=0.9 --opt=sgd --lr=0.1 --loss=crossent --nonlin=relu --nworkers=12 --use-bn --no-val --plot-test --data-root=<path/to/imagenet>

# Thresholded-ReLU activation
python3 train.py --ds=imagenet --arch=resnet18 --batch=256 --wtdecay=1e-4 --lr-decay=0.1 --epochs 90 --lr-decay-epochs 30 60 --momentum=0.9 --opt=sgd --lr=0.1 --loss=crossent --nonlin=threshrelu --nworkers=12 --use-bn --no-val --plot-test --data-root=<path/to/imagenet>

```

## Acknowledgments

This repository includes code from:
* The [PyTorch AlexNet implementation](https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py) adapted based on code from the [DoReFa implementation](https://github.com/ppwwyyxx/tensorpack/blob/master/examples/DoReFa-Net/alexnet-dorefa.py).
* The 8-layer convnet from the [DoReFa repo](https://github.com/ppwwyyxx/tensorpack/blob/master/examples/DoReFa-Net/svhn-digit-dorefa.py).
* The ResNet-18 implementation from PyTorch.
* ConcatDataset from [pytorch/tnt](https://github.com/pytorch/tnt/blob/master/torchnet/dataset/concatdataset.py).
* PartialDataset from [this gist](https://gist.github.com/t-vi/9f6118ff84867e89f3348707c7a1271f) by Thomas Viehmann.
* TensorboardLogger from [this gist](https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514) by Michael Gygli.


