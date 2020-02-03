# Cloud-Retrieval-CNN
Repo to maintain codebase for the CNN model we're developing.

### Questions to be addressed:

* Activation function - Currently, there is an acitvation layer included at the top of the ResNet blocks meaning after each convolution, there will be activation. But in the blocks themselves, there is no BN + activation after the second convolution.

* Output layer - Output(2) is not being computed here

* The `1x2` notation means the feature map is reduced by half. But there is no indication as to how this is done.
