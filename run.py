from diffusers import DDPMPipeline
import os
import torch

# For reproduction
torch.manual_seed(1111)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Importing pretrained DMs
image_pipe = DDPMPipeline.from_pretrained("google/ddpm-ema-celebahq-256")

d = {0:"coarse", 
     1:"medium", 
     2:"fine"}

path = "./results/"
os.makedirs(path,exist_ok=True)
    
print("Generate source image")
process = image_pipe()
images = process['sample'][0]
images.save(os.path.join(path,"source.png"))
condition_s = process['stats']
xT_s = process['xT']
weight = image_pipe.adjust_weight(condition_s)

print("Generate reference image")
process = image_pipe()
images = process['sample'][0]
images.save(os.path.join(path,"reference.png"))
condition_r = process['stats']

print("Start coarse-to-fine Editing")
for criteria in [0,1,2]: # 0: coarse, 1: medium, 2: fine
    print("{} ...".format(d[criteria]))
    process = image_pipe(criteria = criteria, condition = [condition_s, condition_r], xT = xT_s, weight = weight)
    images = process['sample'][0]
    images.save(os.path.join(path,"{}.png".format(d[criteria])))

        
