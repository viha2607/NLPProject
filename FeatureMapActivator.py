class FeatureMapActivator():

    '''
    FeatureMapActivator: is used to activate the feature maps (filters) for given layer by doing forward pass.
    '''

    def __init__(self, layer):
        
        '''
        registers a hook to caculate forward propogation until given layer

        Arguments
        ---------
        layer: (Sequential) selected layer from a cnn model
        '''

        self.hook = layer.register_forward_hook(self.hook_foorward)

    def hook_foorward(self, layer, input, output):
        '''
        carry out forward propogation which saves the leyrs output (feature map)
        '''
        self.featuremap = output

    def close(self):
        '''
        removing hook from memory cache
        '''
        self.hook.remove()