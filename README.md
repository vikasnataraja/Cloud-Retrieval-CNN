loud-Retrieval-CNN
Repo to maintain codebase for the CNN model we're developing.

### Model Architecture
<img src="https://www.mdpi.com/remotesensing/remotesensing-11-01962/article_deploy/html/images/remotesensing-11-01962-g009.png" width="700" height="700" align="middle">

### Questions and Assumpations:

* Activation function - Currently, there is an acitvation layer included at the top of the ResNet blocks meaning after each convolution, there will be activation. But in the blocks themselves, there is no BN + activation after the second convolution. Will need to address this soon enough.

* Output layer - Output(2) is not being computed here

* The `x1/2` notation means the feature map is reduced by half. But there is no indication as to how this is done. Assuming some pooling here and used max pooling.

* The `x8` deconvolution (transposed convolution) does not have any explanation as to how that x8 is achieved. We assume this is bilinear interpolation (will need to address this because this is not very efficient and x8 interpolation is not recommended)

* The 1x1 convolution normally present at the top of the ResNet blocks is absent from this model. Will need to address this.

