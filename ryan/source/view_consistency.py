import torch.nn as nn
import torch
from .unprojector import unproject_translations_individually

epsilon = 1e-2

def weighted_mean(images, weights):
    #Takes BCHW images and BHW weights and returns a pixel-wise CHW average
    
    assert len(images.shape)==4 and len(weights.shape)==3
    
    numerator   = (images*weights[:,None]).sum(dim=0) # CHW tensor
    denominator = weights.sum(dim=0)[None]            # 1HW tensor
    
    denominator[denominator==0] = 1 #Get rid of 0's in denominator
    denominator = denominator + epsilon #The previous line should be enough...but it's not. Alone, puts NAN's in the gradient...technically we don't need the previous line anymore...
    # When a particular pixel has 0 weight in any image, the output pixel is black
    
    return numerator/denominator


def weighted_variance(images, weights):
    #Takes BCHW images and BHW weights and returns a pixel-wise CHW variances
    #Variance is the average squared distance to the mean
    
    mean = weighted_mean(images, weights) #  CHW tensor
    mean = mean[None]                     # 1CHW tensor
    diff = images - mean                  # BCHW tensor
    squared_diff = diff ** 2              # BCHW tensor
    
    numerator   = (squared_diff*weights[:,None]).sum(dim=0) # CHW tensor
    denominator = weights.sum(dim=0)[None]                  # 1HW tensor
    
    denominator[denominator==0] = 1 #Get rid of 0's in denominator.
    denominator = denominator + epsilon #The previous line should be enough...but it's not. Alone, it puts NAN's in the gradient...technically we don't need the previous line anymore...
    
    return numerator/denominator

def normalize_colors(images):
    #TODO: Test this function's correctness in a jupyter notebook demo
    assert len(images.shape)==4, 'images should be BCHW'
    brightness = ((images**2).sum(1,keepdims=True))**.5
    images = images/brightness
    return images

class ViewConsistencyLoss(nn.Module):
    def __init__(self, recovery_width:int=256, recovery_height:int=256, version='std', normalize_colors=True):
        #TODO: add color normalization option
        
        #Usually recovery_width==recovery_height. There's not really a good reason not to make it square
        
        super().__init__()
        
        assert version in ['std','var']
        
        self.version         =version
        self.recovery_width  =recovery_width 
        self.recovery_height =recovery_height
        self.normalize_colors=normalize_colors
        
    def forward(self, scene_translations, scene_uvs, scene_labels):
        
        #Calculate num_labels; it's one less argument we need to specify
        #We're just calculating a loss; it's ok to lose a texture we never see
        num_labels=1+scene_labels.max()
        
        recovered_texture_packs, recovered_weight_packs = unproject_translations_individually(scene_translations  ,
                                                                                              scene_uvs           ,
                                                                                              scene_labels        ,
                                                                                              num_labels          ,
                                                                                              self.recovery_height,
                                                                                              self.recovery_width )
        ##TODO Make this optional
        brightness = ((recovered_texture_packs**2).sum(dim=-3,keepdims=True))**.5
        epsilon=.1#As epsilon goes to 0, we judge more by hue
        brightness += epsilon
        recovered_texture_packs = recovered_texture_packs/brightness

        
        variances=[]
        for i in range(num_labels):
            variances.append(weighted_variance(recovered_texture_packs[:,i],recovered_weight_packs[:,i]))
        variances=torch.stack(variances)

        #The actual value of the output doesn't have much meaning. However, its gradient definitely does.
        if self.version=='var': return torch.mean(variances    )        
        if self.version=='std': return torch.mean(variances**.5)
    
        assert False, 'This line is unreachable' 
