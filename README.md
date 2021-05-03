# Cloud-Retrieval-CNN
Repo to maintain codebase for the CNN model we're developing.

<!--- ### PSPNet-based Model Architecture
<img src="https://github.com/vikasnataraja/Cloud-Retrieval-CNN/blob/master/assets/full_model.png" width="900" height="800" align="middle"> -->


### UNet Model Architecture
<img src="https://github.com/vikasnataraja/Cloud-Retrieval-CNN/blob/master/assets/updated_unet_arch.png" width="900" height="600" align="middle">

<!---

### UNet Results

<img src="https://github.com/vikasnataraja/Cloud-Retrieval-CNN/blob/master/results/unet/unet_output.png" width="900" height="900" align="middle">

<img src="https://github.com/vikasnataraja/Cloud-Retrieval-CNN/blob/master/results/unet/iou.png" width="900" height="600" align="middle">

#### Feature map visualization
<img src="https://github.com/vikasnataraja/Cloud-Retrieval-CNN/blob/master/results/unet/layer_78.png" width="900" height="600" align="middle">

-->

### Setting up the environment

There is a `requirements.txt` file in the root directory and all the packages and libraries necessary for running the model can be installed by using `pip install -r requirements.txt`. If you wish, you can create a virtual environment, our model has been tested with Python 3.6.8 but should work with higher versions as well.

### Training a model

To train the CNN, you will need to run `train.py`. There are multiple command line instructions you can use. For example, the barebones way to run would be:
```
python train.py
```

To run the file and save a model with a name, set batch size and set the number of epochs, you can use:
```
python train.py --model_name mymodelname.h5 --batch_size 16 --epochs 200
```
To view the CLI for this file, use `python train.py --help`, but they are also available below:
```
usage: train.py [-h] [--input_file INPUT_FILE] [--output_file OUTPUT_FILE]
                [--model_dir MODEL_DIR] [--model_name MODEL_NAME]
                [--input_dims INPUT_DIMS] [--num_channels NUM_CHANNELS]
                [--output_dims OUTPUT_DIMS] [--num_classes NUM_CLASSES]
                [--batch_size BATCH_SIZE] [--lr LR] [--epochs EPOCHS]
                [--normalize] [--fine_tune] [--weights_path WEIGHTS_PATH]
                [--augment] [--test_size TEST_SIZE]

optional arguments:
  -h, --help            show this help message and exit
  --input_file INPUT_FILE
                        Path to numpy input images file
  --output_file OUTPUT_FILE
                        Path to numpy ground truth file
  --model_dir MODEL_DIR
                        Directory where model will be saved.If directory does
                        not exist, one will be created
  --model_name MODEL_NAME
                        Model name that will be saved in model_dir
  --input_dims INPUT_DIMS
                        Input dimension
  --num_channels NUM_CHANNELS
                        Number of channels in input images, set to 1 by
                        default
  --output_dims OUTPUT_DIMS
                        Output dimension, set to 64 by default to get 64x64
                        images
  --num_classes NUM_CLASSES
                        Number of classes
  --batch_size BATCH_SIZE
                        Batch size for the model
  --lr LR               Learning rate for the model
  --epochs EPOCHS       Number of epochs to train the model
  --normalize           Pass --normalize to normalize the images. By default,
                        images will not be normalized
  --fine_tune           Pass --fine_tune to load a previous model and fine
                        tune. By default, this is set to False
  --weights_path WEIGHTS_PATH
                        If fine tuning, pass a path to weights that will be
                        loaded and fine-tuned
  --augment             Pass --augment to use data augmentation. By default,
                        no augmentation is used
  --test_size TEST_SIZE
                        Fraction of training image to use for validation
                        during training

```