# Coarse-to-fine_Generation_of_Diffusion
Submitted to TNNLS Journal under the name "**Analyzing Coarse-to-fine Generation of Diffusion Models from the Image Editing Perspective**"

We use diffusers, the link below has details.
🤗 https://github.com/huggingface/diffusers

Here is the file we modified:
- diffusers/models/unet_2d.py
- diffusers/schedulers/scheduling_ddpm.py
- diffusers/pipelines/ddpm/pipeline_ddpm.py

The parts we modified in diffusers are expressed as annotations (TNNLS).

## Requirements

CUDA == 11.1
cudnn == 8.1.0
numpy == 1.19.5
scikit-learn == 0.24.2
torch == 1.8.1+cu111
torchvision == 0.9.1+cu111
diffusers == 0.13.1


## Implementation

python run.py

## Citation

```bibtex
@misc{von-platen-etal-2022-diffusers,
  author = {Patrick von Platen and Suraj Patil and Anton Lozhkov and Pedro Cuenca and Nathan Lambert and Kashif Rasul and Mishig Davaadorj and Thomas Wolf},
  title = {Diffusers: State-of-the-art diffusion models},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/huggingface/diffusers}}
}
```
