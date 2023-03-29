# import python modules
from fastai.conv_learner import *
import torch
import torch.nn as nn
import numpy as np
from cv2 import resize

# import our own classes
import FeatureMapActivator

class FilterVisualizer():

    '''
    FilterVisualizer: is used make necessary computation to visualize the filters for selected model
    '''

    # global varaibles (for description of vars cehck out set_visualizer_params())
    image_size = 56
    upscaling_steps = 12
    upscaling_factor = 1.2
    lr = 1e-1
    weight_decay = 1e-6
    grad_steps = 20
    blur = None

    def __init__(self, selected_model):

        '''
        set CNN model

        Arguments
        ---------
        selected_model: (trochvision.model) selected vision model
        '''
        
        self.selected_model = selected_model

    def set_visualizer_params(self, image_size=56, upscaling_steps=12, upscaling_factor=1.2, lr=1e-1, weight_decay=1e-6, grad_steps=20, blur=5):

        '''
        set other visualizer params to be used in visualizer

        Arguments
        ---------
        image_size: (int) height and width of image for generated noise image
        upscaling_steps: (int) number of times that scaling is done
        upscaling_factor: (float) factor which upscaling is done with 
        lr: (float) learning rate of optimization
        weight_decay: (float) weight decay of optimization
        grad_steps: (int) number of times that gradient descend implemented
        blur: (int) ratio of blur applied in order to reduce high-frequency patterns
        '''

        self.image_size = image_size
        self.upscaling_steps = upscaling_steps
        self.upscaling_factor = upscaling_factor
        self.lr = lr
        self.weight_decay = weight_decay
        self.grad_steps = grad_steps
        self.blur = blur

    def visualize(self, layer, filter):

        '''
        visualises the layer computing the grandient of feature map w.r.t input image

        Arguments
        ---------
        layer: (Sequential) selected layer from a cnn model
        filter: (int) number of kernel filter used to computer feature map
        '''

        # set temporary image size (will be used in scaling)
        image_size_ = self.image_size

        # generate noise image
        img = (np.random.random((self.image_size, self.image_size, 3)) * 20 + 128.)/255.
        # create forward propogation hook for given layer
        featuremapactivator = FeatureMapActivator.FeatureMapActivator(layer)

        # scaling up image upscaling_steps times to have better feature map resolution with optimizing the scaled image each time
        for i in range(self.upscaling_steps):

            # create tranformer for the selected model type
            _, eval_tfms = tfms_from_model(self.selected_model, image_size_)
            # transforom image and makes it gradable
            img_var = V(eval_tfms(img)[None], requires_grad=True) 
            # define Adam optimizer
            optimizer = torch.optim.Adam([img_var], lr=self.lr, weight_decay=self.weight_decay)


            if i > self.upscaling_steps/2:
                grad_steps_ = int(self.grad_steps*1.3)
            else:
                grad_steps_ = self.grad_steps

            # maximize average activation of choosen fature map for given filter w.r.t input image for grad_steps times 
            for n in range(grad_steps_):

                # print(f'up step: {i} grad step: {n}')

                # initialize grad
                optimizer.zero_grad()
                self.model(img_var)

                # do forward progopogation for selected filter in the given layer to create feature map and caculate loss
                loss = -featuremapactivator.featuremap[0, filter].mean()

                # caculate grandient
                loss.backward()
                # make optimization descend
                optimizer.step()

            # inverse transform the optimized image
            img = eval_tfms.denorm(img_var.data.numpy().transpose(0,2,3,1))[0]
            self.output = img

            # upscaling
            # caculate new upscaled image size
            image_size_ = int(self.upscaling_factor * image_size_)
            # scale up image
            img = cv2.resize(img, (image_size_, image_size_), interpolation = cv2.INTER_CUBIC)
            # blur image to reduce high frequency patterns
            if self.blur is not None: img = cv2.blur(img, (self.blur, self.blur))

        # close the generated map to clear up the cache   
        featuremapactivator.close()

        return np.clip(self.output, 0, 1)
    
    def get_transformed_image(self, image, transform_size=224):
        '''
        create transformer for the selected model type, transforms the selected image then denormalize transfrom
        
        Arguments
        ---------
        img: (np.array) image
        transform_size: (int) widith and height of what is used to transform the given image
        '''

        assert (isinstance(image, np.ndarray)), "type of image variable should be numpy.ndarray"

        _, eval_tfms = tfms_from_model(self.selected_model, transform_size)
        return eval_tfms.denorm(eval_tfms(image/255.).transpose(1,2,0))
        
    
    def get_nstrongest_filters(self, image, layer, n, transform_size=224):

        '''
        find the n highest activated filters (feature map) for given image

        Arguments
        ---------
        image: (np.array) image
        transform_size: (int) widith and height of what is used to transform the given image
        layer: (Sequentail) selected layer in CNN model for feature maximization. It will only look for the filter (feature map) of that layer
        n: (int) number of first n highest activated feature map

        Output
        ------
        nstrongest_filters: (int) indices of n stongest filters
        mean_filters_activations: (list) mean activation values of filters

        '''

        assert (isinstance(image, np.ndarray)), "type of image variable should be numpy.ndarray"

        # create transformer for the selected model type
        _, eval_tfms = tfms_from_model(self.selected_model, transform_size)
        tf_image = eval_tfms(image/255.)

        # create forward propogation hook for given layer
        featuremapactivator = FeatureMapActivator.FeatureMapActivator(layer)
        
        # fit the picture to the model
        self.model(V(tf_image)[None])
        
        # sort all the feature maps based on their mean activation in descending order
        mean_filters_activations = [float(featuremapactivator.featuremap[0,i].mean().data.cpu().numpy()) for i in range(featuremapactivator.featuremap.shape[1])]
        nstrongest_filters = sorted(range(len(mean_filters_activations)), key=lambda i: mean_filters_activations[i])[-n:][::-1]
        
        # close the generated map to clear up the cache   
        featuremapactivator.close()

        return nstrongest_filters, mean_filters_activations


    def generate_selected_fmaps(self, layer, filter_list):

        '''
        compute feature map based on noise image for given layer and selected range of filters

        Arguments
        ---------
        layer: (Sequential) selected layer of CNN model
        filter_list: (list) list of selected filter  (e.g., [1,2,3])

        Output
        ------
        fmap: list of calculated fmaps
        name_fmap: name of the fmap filters 
        '''
        assert (type(filter_list)==list), "filter_list must be list object"

        # list to store caculated feautre maps
        fmaps = []
        # list to store name of the fmap filters (will use for the polting purpose)
        name_fmaps = []


        # calculate each feature map and add it to the list
        for filter_i in filter_list:
            fmaps.append(self.visualize(layer, filter_i))

            name_fmaps.append(filter_i)

        return fmaps, name_fmaps