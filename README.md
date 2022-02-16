# Cloud-Retrieval-CNN
Repo to maintain codebase for the CNN model we're developing.


### UNet Model Architecture
<img src="https://github.com/vikasnataraja/Cloud-Retrieval-CNN/blob/master/assets/updated_architecture.png" width="400" height="200" align="middle">


### Setting up the environment

There are 2 ways to setup the packages required to run the model. We recommend using a `virtualenv` or an anaconda environment. 

#### Option 1 - Using pip

There is a `requirements.txt` file in the root directory and all the packages and libraries necessary for running the model can be installed as follows:

```
pip install -r requirements.txt
```

Our model has been tested with Python 3.6.8 but should work with higher versions as well.

#### Option 2 - Using anaconda

If using a conda environment, the `install_packages.sh` will install all the necessary packages. This has been tested on `py >= 3.6`. Use the following command in a bash environemnt:

```
sh install_packages.sh
```

Alternatively, you could use `bash install_packages.sh` as well.


### Training a model

To train the CNN, you will need to run `train.py`. There are multiple command line instructions you can use. For example, the barebones way to run would be:
```
python3 train.py
```

To run the file and save a model with a name, set batch size and set the number of epochs, you can use:
```
python3 train.py --model_name mymodelname.h5 --batch_size 16 --epochs 200
```
To view the CLI for this file, use `python train.py --help`, but they are also available below:
```
usage: train.py [-h] [--radiance_file RADIANCE_FILE] [--cot_file COT_FILE]
                [--model_dir MODEL_DIR] [--model_name MODEL_NAME]
                [--input_dims INPUT_DIMS] [--num_channels NUM_CHANNELS]
                [--output_dims OUTPUT_DIMS] [--num_classes NUM_CLASSES]
                [--batch_size BATCH_SIZE] [--lr LR] [--epochs EPOCHS]
                [--normalize] [--fine_tune] [--weights_path WEIGHTS_PATH]
                [--test_size TEST_SIZE]

optional arguments:
  -h, --help            show this help message and exit
  --radiance_file RADIANCE_FILE
                        Path to numpy input radiance images file
  --cot_file COT_FILE   Path to numpy ground truth COT file
  --model_dir MODEL_DIR
                        Directory where model will be saved. If directory
                        entered does not exist, one will be created
  --model_name MODEL_NAME
                        Model name that will be saved in model_dir
  --input_dims INPUT_DIMS
                        Dimension (width or height) of the input image. Set to
                        64 by default
  --num_channels NUM_CHANNELS
                        Number of channels in the input images, set to 1 by
                        default to use a single wavelength
  --output_dims OUTPUT_DIMS
                        Dimension (width or height) of the output/target COT,
                        set to 64 by default to get 64x64 images
  --num_classes NUM_CLASSES
                        Number of classes. Set to 36 by default
  --batch_size BATCH_SIZE
                        Batch size for the model. Set to 32 by default.
  --lr LR               Learning rate for the model. Set to 0.001 by default
  --epochs EPOCHS       Number of epochs to train the model. Set to 500 by
                        default
  --normalize           Pass --normalize to normalize the images. By default,
                        images will not be normalized
  --fine_tune           Pass --fine_tune to load a previous model and fine
                        tune. By default, this is set to False
  --weights_path WEIGHTS_PATH
                        If fine tuning, pass a path to weights that will be
                        loaded and fine-tuned
  --test_size TEST_SIZE
                        Fraction of training image to use for validation
                        during training. Defaults to using 20% of the data
```
