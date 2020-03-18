# FBA-Matting
Official repository for the paper [**F, B, Alpha Matting**](https://arxiv.org/abs/2003.07711), under review at ECCV2020.  
Marco Forte<sup>1</sup>, François Pitié<sup>1</sup>  

<sup>1</sup> Trinity College Dublin

Matting and compositing results on real images.
<p align="center">
  <img src="xxx" width="640" title="Our results"/>
</p>

## Requirements
#### Packages:
- torch >= 1.4
- numpy
- opencv-python
- scikit-image

GPU memory >= 11GB for inference on Adobe Composition-1K testing set (Resolution above 1920x1080).

## Models
**These models have been trained on Adobe Image Matting Dataset. They are covered by the [Adobe Deep Image Mattng Dataset License Agreement](https://drive.google.com/open?id=1MKRen-TDGXYxm9IawPAZrdXQIYhI0XRf) so they can only be used and distributed for noncommercial purposes.**

| Model Name  |     File Size   | SAD | MSE | Grad | Conn |
| :------------- |------------:| :-----|----:|----:|----:|
| [FBA]() (xxx) Table. xxx      |Adobe Matting Dataset| xxx      |   xxx |xxx|xxx|xxx|


## Prediction 
We provide a script and jupyter notebook for predictions using our model.   

...todo...
