![](images/exmaple_result.png)
### Visualizing Convolutional Neural Network features based on method proposed by [Erhan et al. (2009)](https://www.researchgate.net/publication/265022827_Visualizing_Higher-Layer_Features_of_a_Deep_Network).

This repository contains the implementation for visualizing the deep layer feature by extending Erhan et al. (2009) for deep ConvNets. The main goal of this term project is to have a better interpretation of the high levels feature representations attained by deep ConvNets, as true posterior are intractable for high layers. To that extend, we adapted a gradient-based optimization technique called activation maximization to ConvNets for reconstructing the feature space characterized by the filters in high layers. Using this method we carried out several experiments to demonstrate how to make qualitative inferences about such models in terms of their capability to learn the feature space captured by training samples.

We build visualization from random noise image by updating it along the opposite direction of the gradient of the selected activation with respect to the reconstructed image. Doing that we acquired visualization in which the selected filters in any layer is characterized by maximization the activation of that filter. Even though the nature of the gradient problem is non-convex, this methods still manages to produce sensible visualization for the filter its associated with. The only drawback is that the gradient problems are more sensitive to the choice of hyperparameters and they have to be chosen carefully in order to obtain proper visualization.

In the first stage, a feature map for different layers has been constructed. Then, we later fit the selected picture to the model to compute the forward propagation. The features seem more complicated at higher layers. Based on the result of n most activated filters, then we constructed the feature maps again for these filters at higher layers using the premonition method, to see how the activation of the feature corresponds to the image.

For the experiment, we used pre-trained torchvision.models.resnet34. However, it can easily be extended to the other models. For more detail, please check out the code and the paper.

**note: this work is based on [fg91/visualizing-cnn-feature-maps](https://github.com/fg91/visualizing-cnn-feature-maps).**

#### Files: 
- ```FeatureMapActivator.py```: is used to activate the feature maps (filters) for given layer by doing forward pass.
- ```FilterVisualizer.py```: is used to make necessary computation to visualize the features for selected model
- ```utils.py```: utility functions
- ```experiments.ipynb```: contains experiment we conducted

#### How to run:
```
conda install nb_conda_kernels
conda env create -f env.yml
ipython kernel install --user --name=explain_cnn_filters
```
