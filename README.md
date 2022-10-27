# SOLO_Instance_segmentation (Pytorch Lightning implementation)

The project involved the implementation of the instance segmentation algorithm defined in paper: [SOLO](https://arxiv.org/abs/1912.04488). The main idea behind the paper is that different instance categories can be differentiated by their location and size in an image. The image is divided into SxS grids. Each grid cell is responsible for locating the instance's center which falls in the pixels it corresponds to in the original image. <br>
Size is handled by detecting at different levels of Feature Pyramid Network. Each level of the feature pyramid contributes in producing a prediction. The smallest level of the Feature pyramid network predicts the large objects and their masks while the largest level predicts the small objects. <br>
The Feature pyramid network uses the Resnet50 backbone as the model architecture.

# Network architecture

<img src="./Results/fpn_model.JPG" align = "center">

# Results

Following are the loss curves for training and validation:

