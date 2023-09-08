# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

# limitations under the License.


import torch

from tqdm.auto import tqdm

from ...pipeline_utils import DiffusionPipeline
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from torchvision import transforms as T, utils
def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5
    
def find_nearest(array, value):
    # Find the position of an element close to a specific value in an array (TNNLS)
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
    
class DDPMPipeline(DiffusionPipeline):
    def __init__(self, unet, scheduler):
        super().__init__()
        scheduler = scheduler.set_format("pt")
        self.register_modules(unet=unet, scheduler=scheduler)
        
        # Range Configuration based on Noise Schedules (TNNLS)
        total_T = self.scheduler.timesteps.max()
        alphas_cumprod = self.scheduler.alphas_cumprod
        y1_d = np.r_[0, np.diff(alphas_cumprod)]/0.1
        abd = np.abs(y1_d)
        yq = np.quantile(abd, 0.1)
        ar_idx = np.where(abd>yq)[0].max()
        zero_point = y1_d[1:ar_idx].argmin()
        array = y1_d[zero_point:ar_idx]
        idx1 = find_nearest(array, np.quantile(array, 0.333))
        array = y1_d[1:zero_point]
        idx2 = find_nearest(array, np.quantile(array, 0.333))
        c_l = total_T -  (zero_point+idx1)
        f_l = 1+idx2
        m_l = total_T - (f_l + c_l)
        print("Deterministic coarse-to-fine range: [{}:{}] [{}:{}] [{}:{}]".format( \
                                  total_T, total_T - c_l, \
                                  total_T - c_l - 1, f_l + 1, \
                                  f_l, 0))
        self.total_T = total_T
        self.c_l = total_T-c_l
        self.f_l = f_l
        
    def adjust_weight(self, stats):
        # Make the  \hat{h}(t) (TNNLS)
        # Find the time step point where \hat{h}(t) reaches 0 (TNNLS)
        h_list = []
        for j in stats:
            h_list.append(j[1].detach().cpu().numpy())
        h_list = np.array(h_list).reshape(1000,-1)
        pca = PCA(n_components=2)
        h_pca = pca.fit_transform(h_list)
        hh = []
        for i in h_pca:
            hh.append(i.mean())
        hh = np.array(hh)[::-1]
        turn = 0
        if hh[0] > 0:
            fflag = False
        else:
            fflag = True
        for n,i in enumerate(hh):
            if i > 0:
                flag = False
            else:
                flag = True
            if fflag != flag:
                turn = n
                break
        turn = 1000-turn
        if turn < 558:
            return 0
        w = int(np.abs(turn-558)*(100.0/180.0)) 
        return w    

    @torch.no_grad()
    def __call__(self,criteria=None, condition=None, xT=None, weight = None, batch_size=1, generator=None, torch_device=None, output_type="pil"):
        if torch_device is None:
            torch_device = "cuda" if torch.cuda.is_available() else "cpu"

        self.unet.to(torch_device)

        image = torch.randn(
            (batch_size, self.unet.in_channels, self.unet.sample_size, self.unet.sample_size),
            generator=generator,
        )
        if xT is not None:
            image = xT
        xT = image.clone() # Save for reproduction (TNNLS)
            
        image = image.to(torch_device)
        if condition is not None:
            # Range Adjustment based on Latent Features (TNNLS)
            weight = weight
            total_T = self.total_T
            c_l = self.c_l
            f_l = self.f_l
            print("Adjusted coarse-to-fine range: [{}:{}] [{}:{}] [{}:{}]".format( \
                                  total_T, c_l + weight, \
                                  c_l - 1 + weight, f_l + 1 + weight, \
                                  f_l + weight, 0))
            coarse = c_l + weight
            fine = f_l + weight
        self.scheduler.set_timesteps(1000)
        stats = []
        for t in tqdm(self.scheduler.timesteps):
            if condition is None:
                # Common reverse process (TNNLS)
                model_output = self.unet(image, t)
                h = model_output["h"] # Save for manipulating (TNNLS)
                model_output = model_output["sample"]
                process = self.scheduler.step(model_output, t, image)
            else:
                # ------------------------------------------------------------------------------
                if criteria == 0: # Coarse editing (TNNLS)
                    if t >= coarse:
                        # Manipulating process (TNNLS)
                        model_output = self.unet(image, t, [condition[1][t][1],0.75])
                    else:
                        # Common reverse process (TNNLS)
                        model_output = self.unet(image, t)
                        
                elif criteria == 1: # Medium editing (TNNLS)
                    if t > fine and t < coarse:
                        # Manipulating process (TNNLS)
                        model_output = self.unet(image, t, [condition[1][t][1],0.75])
                    else:
                        # Common reverse process (TNNLS)
                        model_output = self.unet(image, t)
                        
                elif criteria == 2: # Fine editing (TNNLS)
                    if t <= fine and t > 10:
                        # Manipulating process (TNNLS)
                        model_output = self.unet(image, t, [condition[1][t][1],0.75])
                    else:
                        # Common reverse process (TNNLS)
                        model_output = self.unet(image, t)
                h=0
                c = condition[0][t]
                process = self.scheduler.step(model_output["sample"], t, image, condition = c)
            image = process["prev_sample"]
            noise = process["noise"] # Save for reproduction (TNNLS)
            stats.append([noise,h])
            
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)
        stats.reverse()
        return {"sample": image, "stats":stats,"xT":xT}
