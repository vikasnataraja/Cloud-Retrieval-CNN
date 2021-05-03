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
