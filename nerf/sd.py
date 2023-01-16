"""
This code originally came from the Stable-Dreamfusion codebase,

@misc{stable-dreamfusion,
    Author = {Jiaxiang Tang},
    Year = {2022},
    Note = {https://github.com/ashawkey/stable-dreamfusion},
    Title = {Stable-dreamfusion: Text-to-3D with Stable-diffusion}
}

This is the only file from that repository we needed to use (sd.py)
It has been heavily modified to suit Peekaboo better, but the basic concepts are the same
I've also added tensor shape assertions into the code, so it should be easier to read
"""

from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from diffusers import StableDiffusionPipeline

# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

# import time

class StableDiffusion(nn.Module):
    def __init__(self, device, checkpoint_path="CompVis/stable-diffusion-v1-4"):
        super().__init__()

        try:
            with open('./TOKEN', 'r') as f:
                self.token = f.read().replace('\n', '') # remove the last \n!
                print(f'[INFO] loaded hugging face access token from ./TOKEN!')
        except FileNotFoundError:
            self.token = True
            print(f'[INFO] try to load hugging face access token from the default place, make sure you have run `huggingface-cli login`.')
        
        self.device = device
        self.num_train_timesteps = 1000
        self.min_step = int(self.num_train_timesteps * 0.02)
        self.max_step = int(self.num_train_timesteps * 0.98)

        print(f'[INFO] loading stable diffusion...')
        
        #Unlike the original code, I'll load these from the pipeline. This lets us use dreambooth...hopefully...
        pipe = StableDiffusionPipeline.from_pretrained(checkpoint_path, torch_dtype=torch.float)
        pipe.safety_checker = lambda images, a: images, False # Disable Safety Checker (nobody wants it)
    
        pipe.scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=self.num_train_timesteps) #Error from scheduling_lms_discrete.py
        
        self.pipe         = pipe
        self.vae          = pipe.vae         .to(self.device) ; assert isinstance(self.vae          , AutoencoderKL       ),type(self.vae          )
        self.tokenizer    = pipe.tokenizer                    ; assert isinstance(self.tokenizer    , CLIPTokenizer       ),type(self.tokenizer    )
        self.text_encoder = pipe.text_encoder.to(self.device) ; assert isinstance(self.text_encoder , CLIPTextModel       ),type(self.text_encoder )
        self.unet         = pipe.unet        .to(self.device) ; assert isinstance(self.unet         , UNet2DConditionModel),type(self.unet         )
        self.scheduler    = pipe.scheduler                    ; #assert isinstance(self.scheduler    , PNDMScheduler       ),type(self.scheduler    )
        
        self.uncond_text=''
        
                
        # 1. Load the autoencoder model which will be used to decode the latents into image space. 
        # self.vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", use_auth_token=self.token).to(self.device)

        # 2. Load the tokenizer and text encoder to tokenize and encode the text. 
        # self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        # self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)

        # 3. The UNet model for generating the latents.
        self.checkpoint_path=checkpoint_path
        # self.unet = UNet2DConditionModel.from_pretrained(checkpoint_path, subfolder="unet", use_auth_token=self.token).to(self.device)
            

        # 4. Create a scheduler for inference
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience

        print(f'[INFO] loaded stable diffusion!')

    def get_text_embeds(self, prompts):
        
        #THIS IS BAD NAMING! The argument name here should be 'prompts' instead of 'prompt', as it is used in prompt_to_img! This explains why the first dim of the output was all fucky...

        if isinstance(prompts,str):
            prompts=[prompts]

        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompts, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt').input_ids

        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.to(self.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer([self.uncond_text] * len(prompts), padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt').input_ids

        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.to(self.device))[0]

        # Cat for final embeddings
        assert len(uncond_embeddings)==len(text_embeddings)==len(prompts)==len(text_input)==len(uncond_input)

        output_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        assert (uncond_embeddings==torch.stack([uncond_embeddings[0]]*len(uncond_embeddings))).all() #This seems stupidly wasteful...why recompute the embeddings and stack them like that???
        assert (uncond_embeddings==uncond_embeddings[0][None]).all() #This seems stupidly wasteful...why recompute the embeddings and stack them like that???

        assert output_embeddings.shape == (len(prompts)*2, 77, 768)

        return output_embeddings

    def train_step(self, text_embeddings, pred_rgb, guidance_scale=100, t=None,by_loss=False):
        
        # interp to 512x512 to be fed into vae.

        pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)

        if t is None:
            # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
            t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device)

        assert 0<=t<self.num_train_timesteps, 'invalid timestep t=%i'%t

        # encode image into latents with vae, requires grad!
        latents = self.encode_imgs(pred_rgb_512)

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

        # perform guidance (high scale from paper!)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # w(t), sigma_t^2
        w = (1 - self.alphas[t])
        grad = w * (noise_pred - noise)
        #This is equivalent to subtracting the predicted latent images...except 

        # clip grad for stable training?
        # grad = grad.clamp(-1, 1)

        if not by_loss:
            # manually backward, since we omitted an item in grad and cannot simply autodiff.
            latents.backward(gradient=grad, retain_graph=True)
            return 0 # dummy loss value
        else:
            #Ryan's way...
            #WHY THE FUCK DOESN'T THIS WORK? Run some tests to make sure we can get the equivalent of backward this way....I want a LOSS not a thing that inserts gradients...
            with torch.no_grad():
                target=latents+grad
            loss= (target-latents).sum()
            # loss=-loss #Flip it around because optimizers like to decrease loss...and we already know the gradients...
            return loss



    def produce_latents(self, text_embeddings, height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):

        #text_embeddings is a 3d tensor; I'm not sure what the dims mean. But the first one changes with the length of the prompt, and isn't always an even number.
        #First dim is batch_size * 2 (which is weird - I feel like that should be a tuple or something?)

        assert len(text_embeddings.shape)==3 and text_embeddings.shape[-2:]==(77,768)
        num_prompts = len(text_embeddings)//2

        if latents is None:
            latents = torch.randn((num_prompts, self.unet.in_channels, height // 8, width // 8), device=self.device)

        # with torch.no_grad():
        #     white_image = np.ones((1, 3, height, width), dtype=np.float32)
        #     white_image = torch.tensor(white_image).to(self.device)
        #     white_latent = self.encode_imgs(white_image)

        assert 0 <= num_inference_steps <= 1000, 'Stable diffusion appears to be trained with 1000 timesteps'

        self.scheduler.set_timesteps(num_inference_steps)

        with torch.autocast('cuda'):
            for i, t in enumerate(self.scheduler.timesteps):
                assert int(t) == t and 0 <= t <= 999, 'Suprisingly to me...the timesteps were encoded as integers lol (np.int64)'
                assert int(i) == i and 0 <= i <= 999, 'And because there are 1000 of them, the index is also bounded'
                t=int(t) #Makes some schedulers happy; it's the same value anyway

                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([latents] * 2) #The first half is the blank prompts (repeated); the second half is 

                # predict the noise r['sample']
                with torch.no_grad():
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']
                    assert len(latent_model_input)==len(text_embeddings)==len(noise_pred)

                # perform guidance
                # print(noise_pred.shape)#torch.Size([2, 4, 64, 64])
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                # print(noise_pred.shape)#torch.Size([1, 4, 64, 64])

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents)['prev_sample'] #It's a dict with nothing but 'prev_sample'...why?!? This is silly??
                # print('Latents Shape:',latents.shape) #torch.Size([1, 4, 64, 64]) (regardless of prompt size)

                # clean_latents = self.scheduler.step(noise_pred, 0, latents)['prev_sample']
                # diff_latents       = white_latent-clean_latents
                # K=100
                # latents[:,:,0,:]  += diff_latents[:,:,0,:]/K #Add white edge
                # latents[:,:,-1,:] += diff_latents[:,:,-1,:]/K #Add white edge
                # latents[:,:,:,0]  += diff_latents[:,:,:,0]/K #Add white edge
                # latents[:,:,:,-1] += diff_latents[:,:,:,-1]/K #Add white edge
        
        return latents

    def decode_latents(self, latents):

        latents = 1 / 0.18215 * latents

        with torch.no_grad():
            # imgs = self.vae.decode(latents).sample
            imgs = self.vae.decode(latents)
            if hasattr(imgs,'sample'):
                imgs=imgs.sample #If isinstnace(imgs,DecoderOutput) - unlike in rlab

        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        
        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        #Original, broken code:
        # posterior = self.vae.encode(imgs).latent_dist
        # latents = posterior.sample() * 0.18215

        posterior = self.vae.encode(imgs)
        latents = posterior.sample() * 0.18215

        return latents

    
    def embed_to_img(self, text_embeds, height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):

        # if isinstance(prompts, str):
            # prompts = [prompts]

        # Prompts -> text embeds
        # text_embeds = self.get_text_embeds(prompts) # [2, 77, 768]
        # assert text_embeds.shape==( len(prompts)+1, 77, 768 )

        # Text embeds -> img latents
        latents = self.produce_latents(text_embeds, height=height, width=width, latents=latents, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale) # [1, 4, 64, 64]
        
        # Img latents -> imgs
        imgs = self.decode_latents(latents) # [1, 3, 512, 512]

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        # imgs = (imgs * 255).round().astype('uint8') #We don't need this...rp can handle float images without any trouble...

        return imgs


    
    def prompt_to_img(self, prompts, height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):

        if isinstance(prompts, str):
            prompts = [prompts]

        # Prompts -> text embeds
        text_embeds = self.get_text_embeds(prompts) # [2, 77, 768]
        assert text_embeds.shape==( len(prompts)+1, 77, 768 )
        return self.embed_to_img(text_embeds, height ,width ,num_inference_steps ,guidance_scale ,latents)

#         # Text embeds -> img latents
#         latents = self.produce_latents(text_embeds, height=height, width=width, latents=latents, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale) # [1, 4, 64, 64]
        
#         # Img latents -> imgs
#         imgs = self.decode_latents(latents) # [1, 3, 512, 512]

#         # Img to Numpy
#         imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
#         # imgs = (imgs * 255).round().astype('uint8') #We don't need this...rp can handle float images without any trouble...

#         return imgs


if __name__ == '__main__':

    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('prompt', type=str)
    parser.add_argument('-H', type=int, default=512)
    parser.add_argument('-W', type=int, default=512)
    parser.add_argument('--steps', type=int, default=50)
    opt = parser.parse_args()

    device = torch.device('cuda')

    sd = StableDiffusion(device)

    imgs = sd.prompt_to_img(opt.prompt, opt.H, opt.W, opt.steps)

    # visualize image
    plt.imshow(imgs[0])
    plt.show()




