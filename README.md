# Cloud-Retrieval-CNN
Repo to maintain codebase for the CNN model we're developing.


### PSPNet-based Model Architecture
<img src="https://github.com/vikasnataraja/Cloud-Retrieval-CNN/blob/master/extras/full_model.png" width="900" height="800" align="middle">


### UNet Model Architecture
<img src="https://github.com/vikasnataraja/Cloud-Retrieval-CNN/blob/master/extras/u-net-architecture.png" width="900" height="600" align="middle">


### Notes about the original paper (Masuda et al.):

* Output layer - Output(2) is not being computed here

* The `x8` deconvolution (transposed convolution) does not have any explanation as to how that x8 is achieved. We assume this is bilinear interpolation (will need to address this because this is not very efficient and x8 interpolation is not recommended)

* The 1x1 convolution normally present at the top of the ResNet blocks is absent from this model. Will need to address this.

### Notes about UNet

* There is an option to switch the transposed convolution + upsampling blocks with convolution + upsampling blocks. We are using the former as it performs better with our dataset.

* Need a bigger image size to try valid padding.

* Need to try `Cropping2D` before concatenation like the paper suggests.
