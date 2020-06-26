# Cloud-Retrieval-CNN
Repo to maintain codebase for the CNN model we're developing.


### PSPNet-based Model Architecture
<img src="https://github.com/vikasnataraja/Cloud-Retrieval-CNN/blob/master/assets/full_model.png" width="900" height="800" align="middle">


### UNet Model Architecture
<img src="https://github.com/vikasnataraja/Cloud-Retrieval-CNN/blob/master/assets/updated_unet_arch.png" width="900" height="600" align="middle">

### UNet Results
<img src="https://github.com/vikasnataraja/Cloud-Retrieval-CNN/blob/master/results/unet/unet_output.png" width="900" height="900" align="middle">

<img src="https://github.com/vikasnataraja/Cloud-Retrieval-CNN/blob/master/results/unet/iou.png" width="900" height="600" align="middle">

#### Feature map visualization
<img src="https://github.com/vikasnataraja/Cloud-Retrieval-CNN/blob/master/results/unet/layer_78.png" width="900" height="600" align="middle">

### Setting up the environment

There is a `requirements.txt` file in the root directory and all the packages and libraries necessary for running the model can be installed by using `pip install -r requirements.txt`. If you wish, you can create a virtual environment, our model has been tested with Python 3.6.8 but should work with higher versions as well.

### Notes about the original paper (Masuda et al.):

* Output layer - Output(2) is not being computed here

* The `x8` deconvolution (transposed convolution) does not have any explanation as to how that x8 is achieved. We assume this is bilinear interpolation (will need to address this because this is not very efficient and x8 interpolation is not recommended)

* The 1x1 convolution normally present at the top of the ResNet blocks is absent from this model. Will need to address this.

### Notes about UNet

* There is an option to switch the transposed convolution + upsampling blocks with convolution + upsampling blocks. We are using the former as it performs better with our dataset.

* Need a bigger image size to try valid padding.

* Need to try `Cropping2D` before concatenation like the paper suggests but will need a 200x200 image for that.

