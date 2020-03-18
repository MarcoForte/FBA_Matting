# FBA-Matting
Official repository for the paper [**F, B, Alpha Matting**](https://arxiv.org/abs/2003.07711), under review at ECCV2020.  
Marco Forte<sup>1</sup>, [François Pitié](https://francois.pitie.net/)<sup>1</sup>  

<sup>1</sup> Trinity College Dublin

Matting and compositing results on real images.
<p align="center">
  <img src="./examples/example_results.png" width="640" title="Our results"/>
</p>

## Requirements
GPU memory >= 11GB for inference on Adobe Composition-1K testing set, more generally for resolutions above 1920x1080.

#### Packages:
- torch >= 1.4
- numpy
- opencv-python
#### Additional Packages for jupyter notebook
- matplotlib
- gdown (to download model inside notebook)


## Models
**These models have been trained on Adobe Image Matting Dataset. They are covered by the [Adobe Deep Image Mattng Dataset License Agreement](https://drive.google.com/open?id=1MKRen-TDGXYxm9IawPAZrdXQIYhI0XRf) so they can only be used and distributed for noncommercial purposes.**

| Model Name  |     File Size   | SAD | MSE | Grad | Conn |
| :------------- |------------:| :-----|----:|----:|----:|
| [FBA](https://drive.google.com/file/d/1T_oiKDE_biWf2kqexMEN7ObWqtXAzbB1/view?usp=sharing) Table. 4  | 139mb | 26.4 | 5.4 | 10.6 | 21.5 |


## Prediction 
We provide a script `demo.py` and jupyter notebook which both give the foreground, background and alpha predictions of our model. 


## Citation

```
@article{forte2020fbamatting,
  title   = {F, B, Alpha Matting},
  author  = {Marco Forte and François Pitié},
  journal = {CoRR},
  volume  = {abs/2003.07711},
  year    = {2020},
}
```