# Cloud-Retrieval-CNN
This repository contains the code that was developed for the paper: ["Segmentation-Based Multi-Pixel Cloud Optical Thickness Retrieval Using a Convolutional Neural Network"](https://amt.copernicus.org/articles/15/5181/2022/amt-15-5181-2022.html). Due to file size limits on Github, the data and the best trained model are available separately at the following links:

HDF5 Data Files: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7008103.svg)](https://doi.org/10.5281/zenodo.7008103)

Trained Model: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7013101.svg)](https://doi.org/10.5281/zenodo.7013101)


Please cite the paper as follows:

Nataraja, V., Schmidt, S., Chen, H., Yamaguchi, T., Kazil, J., Feingold, G., Wolf, K., and Iwabuchi, H.: Segmentation-based multi-pixel cloud optical thickness retrieval using a convolutional neural network, Atmos. Meas. Tech., 15, 5181–5205, https://doi.org/10.5194/amt-15-5181-2022, 2022.

### UNet Model Architecture
<img src="https://github.com/vikasnataraja/Cloud-Retrieval-CNN/blob/master/assets/updated_architecture.png" width="400" height="200" align="middle">


### How do I use this repository?
There are 4 steps to get you started with our CNN:

```
Step 1: Cloning the repository
Step 2: Installing the packages that are pre-requisites for running the CNN
Step 3: Creating your own training data from HDF5 files
Step 4: Train the CNN with the training data you generated
```

### Step 1 - Clone the repo
To clone the repo, use the following command line instructions (CLI) in a directory of your choice:
```
git clone git@github.com:vikasnataraja/Cloud-Retrieval-CNN.git
```

This will create a new folder called `Cloud-Retrieval-CNN`. Enter it using `cd Cloud-Retrieval-CNN`.

### Step 2 - Setting up the environment to install packages

There are 2 ways to setup the packages required to run the model. We recommend using a `virtualenv` or an anaconda environment. 

#### Option 1 - Using pip

There is a `requirements.txt` file in the root directory and all the packages and libraries necessary for running the model can be installed as follows:

```
pip install -r requirements.txt
```

Our model has been tested with Python 3.6.8 but should work with higher versions as well. One caveat is that Python 3.9 is currently unsupported with our repository as it is still new and incompatibile with some packages used in this repo. We recommend using Python 3.6.8 or 3.6.9.

#### Option 2 - Using anaconda

If using a conda environment, the `install_packages.sh` will install all the necessary packages. This has been tested on `py >= 3.6`. Use the following command in a bash environemnt:

```
sh install_packages.sh
```

Alternatively, you could use `bash install_packages.sh` as well.

### Step 3 - Creating data
Note: The HDF5 data files are not included in this repository. The original 6 scenes containing the radiance and cloud optical thickness data from the Sulu Sea LES are available here: https://doi.org/10.5281/zenodo.7008103. The data can be downloaded to a directory of your choice within your system.

To create the training data, `create_npy_data.py` must be used which will create a directory containing 3 files:
`inp_radiance.npy`, `out_cot_3d.npy`, `out_cot_1d.npy`. They contain data of the radiance, true 3D COT and IPA COT respectively.

Run the file as follows:
```
python3 create_npy_data.py --fdir path/to/dir/containing/hdf5/files --dest path/to/a/directory/for/writing/npy/files
```

If the directory entered in `--dest` does not exist, one will be created. Use `python3 create_npy_data.py --help` for more details. After running the python file, check the destination directory to ensure that the 3 `npy` files were created.

### Step 4 - Training a model

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

The best trained model is available here: https://doi.org/10.5281/zenodo.7013100
