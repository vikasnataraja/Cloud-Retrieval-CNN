# Cloud-Retrieval-CNN
Repo to maintain codebase for the CNN model we're developing.

### PSPNet-based Model Architecture
<img src="https://www.mdpi.com/remotesensing/remotesensing-11-01962/article_deploy/html/images/remotesensing-11-01962-g009.png" width="700" height="700" align="middle">

### UNet Model Architecture
<img src="https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png" width="700" height="400" align="middle">


### Notes about the original paper (Masuda et al.):

* Output layer - Output(2) is not being computed here

* The `x8` deconvolution (transposed convolution) does not have any explanation as to how that x8 is achieved. We assume this is bilinear interpolation (will need to address this because this is not very efficient and x8 interpolation is not recommended)

* The 1x1 convolution normally present at the top of the ResNet blocks is absent from this model. Will need to address this.

### Notes about UNet

* There is an option to switch the transposed convolution + upsampling blocks with convolution + upsampling blocks. We are using the former as it performs better with our dataset.

* Need a bigger image size to try valid padding.

* Need to try `Cropping2D` before concatenation like the paper suggests.
