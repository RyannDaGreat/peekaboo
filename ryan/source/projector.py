import einops
import torch


def colorized_scene_labels(scene_labels, label_colors):
    #Takes scene_labels, and turns each label into a certain color
    #This is useful for visualization, but might later be used for learning purposes as well
    #This is a pure function, as it does not mutate any of the inputs
    
    #------- Inputs Validation -------
    
    assert isinstance(scene_labels,torch.Tensor)
    assert isinstance(label_colors,torch.Tensor)
    
    batch_size, scene_height, scene_width = scene_labels.shape
    num_labels, num_channels              = label_colors.shape
    
    assert not scene_labels.dtype.is_floating_point
    assert     scene_labels.min()>=0 and scene_labels.max()<num_labels
    

    #------- Output Calculation -------
    
    output = einops.rearrange(scene_labels, 'BS SH SW -> (BS SH SW)')
    output = label_colors[output]
    output = einops.rearrange(output, '(BS SH SW) NC -> BS NC SH SW',
                              BS=batch_size                         ,
                              SH=scene_height                       ,
                              SW=scene_width                        )

    
    #------- Output Validation -------
    
    assert output.shape == (batch_size, num_channels, scene_height, scene_width)
    
    assert output.dtype == label_colors.dtype
    
    return output


def project_textures(scene_uvs,scene_labels,textures):
    
    #TODO: Add linear interpolation like in unproject. This can be done.
    
    #------- Inputs Validation -------
    
    batch_size, scene_height, scene_width = scene_labels.shape
    assert not scene_labels.dtype.is_floating_point
    
    assert scene_uvs.shape == (batch_size, 2, scene_height, scene_width)
    
    num_labels, num_channels, texture_height, texture_width = textures.shape
    assert scene_labels.min()>=0 and scene_labels.max() < num_labels
    
    
    #------- Output Calculation -------
    
    u,v = einops.rearrange(scene_uvs, 'BS UV SH SW -> UV BS SH SW')
    u=torch.clamp(u*texture_height, min=0,max=texture_height-1).long()
    v=torch.clamp(v*texture_width , min=0,max=texture_width -1).long()
    
    assert u.shape == v.shape == (batch_size, scene_height, scene_width)
    
    textures     = einops.rearrange(textures, 'NL NC TH TW -> NL TH TW NC')
    
    scene_labels = einops.rearrange(scene_labels, 'BS SW SH -> (BS SW SH)')
    u            = einops.rearrange(u           , 'BS SW SH -> (BS SW SH)')
    v            = einops.rearrange(v           , 'BS SW SH -> (BS SW SH)')
                                    
    output = textures[scene_labels, u, v]
    output = einops.rearrange(output, '(BS SH SW) NC -> BS NC SH SW',
                              BS=batch_size                         ,
                              SH=scene_height                       ,
                              SW=scene_width                        )
    
    
    #------- Output Validation -------
    
    assert output.shape == (batch_size, num_channels, scene_height, scene_width)

    return output