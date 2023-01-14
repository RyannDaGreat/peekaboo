import einops
import torch


def calculate_subpixel_weights(x,y):
    #This function's math is explained here: 
    #   https://www.desmos.com/calculator/esool5qrrd

    Rx=x%1
    Ry=y%1
    Qx=1-Rx
    Qy=1-Ry

    A=Rx*Ry #Weight for  ceil(x), ceil(y)
    B=Rx*Qy #Weight for  ceil(x),floor(y)
    C=Qx*Qy #Weight for floor(x),floor(x)
    D=Qx*Ry #Weight for floor(x), ceil(y)

    Cx=x.ceil ().long()
    Cy=y.ceil ().long()
    Fx=x.floor().long()
    Fy=y.floor().long()

    X=torch.stack((Cx,Cx,Fx,Fx)) #All X values
    Y=torch.stack((Cy,Fy,Fy,Cy)) #All Y values
    W=torch.stack((A ,B ,C ,D )) #Weights

    return X,Y,W


def unproject_translations(scene_translations : torch.Tensor,
                           scene_uvs          : torch.Tensor,
                           scene_labels       : torch.Tensor,
                           num_labels         : int         ,
                           output_height      : int         ,
                           output_width       : int         ,
                           version            : str='fast'  ):
    #This function is the opposite of the project_textures function
    #It's an inverse function, in a way
    #See the unproject_translations_demo.ipynb for a visual example
    #
    #This is a pure function: it doesn't mutate any of the input tensors
    #It outputs two torch.Tensor values, one of which is the mean and the other is weights

    #Ryan's personal notes:
    # TODO: Possibly change all things called 'output' to 'texture' to make it consistent naming with other functions
    # TODO: Come up with a better name for this function
    # MAYBE todo: eliminate batch size, because we're going to be doing this once for every image anyway...
    # Batch size is technically unnesecary...but it might be nice to keep it, idk...it combines confidence of multiple 
    #    ... scenes very nicely...(ok, I'll keep batch size)
    # TODO: Perhaps keep track of output_sum_of_squares as well (another vector) so that we could output the variance at
    #    ... the same time as we output the sum?
    # The alternative is running this funtion multiple times, and comparing the texture outputs...which is also a totally 
    #    ... valid approach. It doesn't need BS to do that though...

    #------- Input Validation -------
    
    #Validate scene_uvs
    batch_size, two, scene_height, scene_width = scene_uvs.shape
    assert two == 2, "scene_uvs should have exactly two channels"
    assert scene_uvs.min() >= 0 and scene_uvs.max() <=1, 'All u,v values should be between 0 and 1 (not %.4f and %.4f)' % \
          (scene_uvs.min(),         scene_uvs.max())

    #Validate scene_labels
    assert     scene_labels.shape == (batch_size, scene_height, scene_width)
    assert     scene_labels.max() < num_labels     , "The scene_labels includes labels that can't be included in the output!"
    assert not scene_labels.dtype.is_floating_point, "scene_labels should be an integer type, because it is used as an index"

    #Validate scene_translations
    num_channels = scene_translations.shape[1]
    assert scene_translations.shape == (batch_size, num_channels, scene_height, scene_width)

    #Validate integer arguments
    assert num_labels    >= 1
    assert output_height >= 1
    assert output_width  >= 1

    #Validate version
    assert version in ['fast','slow','low memory']


    #------- Outputs Calculation -------

    #This 'Outputs Calculation' section is acronym-heavy.
    #Acronym Key:
    #  - BS = batch_size
    #  - SH,SW,OH,OW = scene_height,scene_width,output_height,output_width
    #  - NC,NL = num_channels,num_labels
    #  - ABCD = 4: it refers to the four A,B,C,D values in the function calculate_subpixel_weights
    #  - w = weights
    #  - t = weighted scene_translations, which are summed together to get the output
    NC = num_channels
    NL = num_labels
    OH = output_height
    OW = output_width

    #Populate the outputs
    u,v = einops.rearrange(scene_uvs.clone(),'BS NC SH SW -> NC BS SH SW')
    assert u.shape == v.shape == (batch_size, scene_height, scene_width)
    
    #Convert u,v from range [0,1),[0,1) to [0,output_height),[0,output_width)
    #To subtract 1 or not...depends on the projection function, and whether not u=.999999 means a bottom pixel
    u *= output_height#-1
    v *= output_width #-1

    #Calculate the subpixel weights w, and make u,v integers
    u,v,w = calculate_subpixel_weights(u,v)
    assert u.shape == v.shape == w.shape == (4, batch_size, scene_height, scene_width)
    assert not u.dtype.is_floating_point and not v.dtype.is_floating_point

    #The calculate_subpixel_weights is likely to reference indices one index outside of the image.
    #Correct this by pushing those indices back into the image again.
    u=u.clamp(0,output_height-1)
    v=v.clamp(0,output_width -1)
    
    l = scene_labels
    t = scene_translations

    #Reshape stuff
    u = einops.rearrange(u, 'ABCD BS    SH SW -> (BS SH SW ABCD)   '       )
    v = einops.rearrange(v, 'ABCD BS    SH SW -> (BS SH SW ABCD)   '       )
    w = einops.rearrange(w, 'ABCD BS    SH SW -> (BS SH SW ABCD)   '       )
    l = einops.repeat   (l, '     BS    SH SW -> (BS SH SW ABCD)   ',ABCD=4)
    t = einops.repeat   (t, '     BS NC SH SW -> (BS SH SW ABCD) NC',ABCD=4)

    #Create empty output tensors
    output_device = scene_translations.device
    output_weight = torch.zeros((num_labels, output_height, output_width              ), device = output_device)
    output_sum    = torch.zeros((num_labels, output_height, output_width, num_channels), device = output_device)

    assert len(t) == len(u) == len(v) == len(w) == len(l)

    if version=='fast':
        #THE FAST VERSION:
        #This version is very fast, but a bit cryptic
        #
        #Summary of this version using imprecise notation:        
        #  #What I'd like to write, but can't:
        #     output_sum   [l, u, v] += t 
        #     output_weight[l, u, v] += w 
        #     # This doesn't accumulate properly when we have duplicate l,u,v's
        #  #Splitting it into R,G,B:
        #     output_sum   [l, u, v, 0] += t[:,0] 
        #     output_sum   [l, u, v, 1] += t[:,1] 
        #     output_sum   [l, u, v, 2] += t[:,2] 
        #     output_weight[l, u, v] += w 

        output_sum    = output_sum   .view(NL*OH*OW*NC)
        output_weight = output_weight.view(NL*OH*OW   )

        for i in range(NC):
            output_sum.index_add_(0, (l*OH*OW*NC + u*OW*NC + v*NC + i), t[:,i]*w)
        output_weight .index_add_(0, (l*OH*OW    + u*OW    + v       ),        w)
        
        output_sum    = output_sum   .view(NL,OH,OW,NC)
        output_weight = output_weight.view(NL,OH,OW   )

    if version=='low memory':
        #This version is slower than 'fast' but uses less memory
        #TODO: Benchmark speed and memory differences quantitatively
        #It uses less memory with rspect to the batch size
        generator = unproject_translations_individually_generator(scene_translations,
                                                                  scene_uvs         ,
                                                                  scene_labels      ,
                                                                  num_labels        ,
                                                                  output_height     ,
                                                                  output_width      ,
                                                                  version = 'fast'  )

        output_mean, output_weight = combine_individual_unprojections(generator)

        assert output_mean  .shape == (num_labels, num_channels, output_height, output_width)
        assert output_weight.shape == (num_labels,               output_height, output_width)

        return output_mean, output_weight

    if version=='slow':
        #THE SUPER-SLOW VERSION:    
        # This version is very slow, but much easier to understand. 
        # It gives exactly the same results as when version=='fast'
        # This version was used to verify the correctness of the 'fast' version
        # In my experiments, version=='slow' took 181 seconds,
        #     whereas 'fast' took 0.003057 seconds. That's a 59208x speedup!
        # Note: the first call of the 'fast' version often takes longer than
        #      subsequent calls; going from .1 seconds to .003
        for l_,u_,v_,t_,w_ in zip(l,u,v,t,w):
            output_sum   [l_,u_,v_]+=w_*t_
            output_weight[l_,u_,v_]+=w_
        
    output_sum = einops.rearrange(output_sum, 'NL OH OW NC -> NL NC OH OW')
    assert output_sum.shape == (num_labels, num_channels, output_height, output_width)
    
    denominator = output_weight.clone()
    denominator[denominator == 0] = 1  # Avoid division by zero errors
    denominator = denominator[:, None, :, :]  # Make this tensor fit

    output_mean = output_sum/denominator


    #------- Outputs Validation -------

    assert output_mean  .shape == (num_labels, num_channels, output_height, output_width)
    assert output_weight.shape == (num_labels,               output_height, output_width)
    
    return output_mean, output_weight


def unproject_translations_individually(scene_translations : torch.Tensor,
                                        scene_uvs          : torch.Tensor,
                                        scene_labels       : torch.Tensor,
                                        num_labels         : int         ,
                                        output_height      : int         ,
                                        output_width       : int         ,
                                        version            : str='fast'  ):
    #Takes in and spits out the same tensor shapes as unproject_translations(...), except BS is at the beginning
    #That is to say, this outputs (textures,weights) aka (BS NL NC TH TW,  BS NC TH TW) tensors
    #(See the unproject_translations(...) function to see what those two-letter dimension acronyms mean)
    #However, this version applies unproject_translations(...) to each element of the batch individually
    #
    #NOTE: This function shares redundant code with unproject_translations_individually_generator
    #This function COULD be made to USE unproject_translations_individually_generator to generate its output
    #HOWEVER, I'm not sure if unproject_translations_individually_generator (which is used to save memory in unproject_translations's 'low memory' version) actually saves memory
    #So, I don't want to commit to that change...yet.
    
    output_textures=[]
    output_weights =[]
    
    for scene_translation, scene_uv, scene_label in zip(scene_translations, scene_uvs, scene_labels):
    
        texture, weight = unproject_translations(scene_translation[None],
                                                 scene_uv         [None],
                                                 scene_label      [None],
                                                 num_labels             ,
                                                 output_height          ,
                                                 output_width           )
        output_textures.append(texture)
        output_weights .append(weight )
        
    output_textures=torch.stack(output_textures,dim=0)
    output_weights =torch.stack(output_weights ,dim=0)
    
    return output_textures, output_weights


def combine_individual_unprojections(textures_and_weights_generator):
    #TODO: Make this even more efficient (combine it with unproject_translations_individually so it doesn't need to remember them all at once, going from O(n) to O(1) space. Make this a mode of unproject_translations)
    # Returns the same shapes and types as unproject_translations, but takes less vram to compute
    #
    # When using unproject_translations on a large number of frames (like 1000 frames), it will run out of vram
    # To recover a texture from this many frames, use combine_individual_unprojections(unproject_translations_individually(...))
    # Instead of unproject_translations(...)
    #
    #textures and weights can be generators to save memory (instead of having a batch dimension)

    # output_weights = weights.sum(0)[:,None]+.01
    # output_textures = (textures*weights[:,:,None]).sum(0)/(output_weights)
    # return output_textures, output_weights

    output_texture = None
    output_weight  = None

    for texture, weight in textures_and_weights_generator:
        assert len(weight .shape)==3, 'Should be [NL, TH, HW]: '    +str(weight .shape)
        assert len(texture.shape)==4, 'Should be [NL, NC, TH, HW]: '+str(texture.shape)

        if output_texture is None:
            #Initialize the outputs
            output_texture=torch.zeros_like(texture)
            output_weight =torch.zeros_like(weight )

        output_texture += texture * weight[:,None]
        output_weight  += weight

    assert output_texture is not None and output_weight is not None, 'Please give more than 0 textures and weights!'

    output_texture /= output_weight[:,None] + .01

    assert len(output_weight .shape)==3, 'Should be [NL, TH, HW]: '    +str(output_weight .shape)
    assert len(output_texture.shape)==4, 'Should be [NL, NC, TH, HW]: '+str(output_texture.shape)

    return output_texture, output_weight


def unproject_translations_individually_generator(scene_translations : torch.Tensor,
                                                  scene_uvs          : torch.Tensor,
                                                  scene_labels       : torch.Tensor,
                                                  num_labels         : int         ,
                                                  output_height      : int         ,
                                                  output_width       : int         ,
                                                  version            : str='fast'  ):
    #The return_generator option is used to save memory, instead of returning a giant tensor. It's good to pair that with
    #   scene_translations, scene_uvs, and scene_labels being generators too. When return_generator is True, this function 
    #   returns a generator that yields texture, weight pairs instead of two textures,weights tensors.

    assert version!='low memory', 'Cant use the "low_memory" version in unproject_translations in unproject_translations_individually_generator, because that would recurse infinitely'
    
    for scene_translation, scene_uv, scene_label in zip(scene_translations, scene_uvs, scene_labels):
    
        texture, weight = unproject_translations(scene_translation[None],
                                                 scene_uv         [None],
                                                 scene_label      [None],
                                                 num_labels             ,
                                                 output_height          ,
                                                 output_width           )
        yield texture, weight


def get_label_averge_colors(scene_images: torch.Tensor, scene_labels: torch.Tensor, num_labels:int):
    #Retuns a 2d tensor of colors with shape (len(label_values), scene_images.shape[1])
    #This function is the natural opposite of projector.colorized_scene_labels(...)
    #It takes photos or translations, and uses their labels to extract average colors
    #If for some reason none of the labels contain a given label value, instead of returning NaN, it returns black
    
    
    #------- Inputs Validation -------
    
    assert len(scene_images.shape) == 4, 'images should be in the form BCHW'
    assert len(scene_labels.shape) == 3, 'labels should be in the form BHW'
    
    assert scene_labels.shape[0] == scene_images.shape[0], 'images and labels should have same batch size'
    batch_size = len(scene_labels)
    
    assert scene_labels.shape[1] == scene_images.shape[2], 'images and labels should have same height'
    assert scene_labels.shape[2] == scene_images.shape[3], 'images and labels should have same width'
    height, width = scene_labels.shape[1:]
    
    num_channels = scene_images.shape[1]
    
    scene_labels = einops.repeat(scene_labels, 'B H W -> B C H W', C=num_channels)
    
    assert scene_labels.shape == scene_images.shape == (batch_size, num_channels, height, width)
    
    
    #------- Outputs Calculation -------
    
    output = []
    for label in range(num_labels):
        mask = scene_labels==label
        color_sum = (scene_images * mask).sum(dim=(0,2,3))
        assert color_sum.shape == (num_channels, )
        color_mean = color_sum / mask.sum(dim=(0,2,3))
        output.append(color_mean)

        
    output = torch.stack(output)
    
    #If any labels weren't present, then their mask.sum will be 0, as will their color_sum
    #That would make their mean_sum a NaN (aka color_sum/mask.sum aka 0/0) 
    #By default, we will return black
    output = output.nan_to_num(0)
        
    #------- Outputs Validation -------
    
    assert output.shape==(num_labels, num_channels)
    
    return output