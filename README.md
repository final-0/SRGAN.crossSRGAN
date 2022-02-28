# crossSRGAN

## Abstract
crossSRGAN can generate better realistic images than SRGAN under certain condition.
The loss function of crossSRGAN is different from that of SRGAN.
This is because the generator and the discriminator have a more adversarial relationship.

<table>
   <tr>
    <td><img src="images/input.png" width=400 height=400></td>
    <td><img src="images/ground.png" width=400 height=400></td>
   </tr>
   <tr>
    <td align="center">input</td>
    <td align="center">ground truth</td>
   </tr>
   <tr>
    <td><img src="images/output_crossSRGAN.png" width=400 height=400></td>
    <td><img src="images/output_SRGAN.png" width=400 height=400></td>
   </tr>
   <tr>
    <td align="center">output crossSRGAN</td>
    <td align="center">output SRGAN</td>
   </tr>
  </table>

## Dataset Preparation <br>
dataset : https://www.kaggle.com/balraj98/deepglobe-road-extraction-dataset?select=train
- recommendation <br>
&emsp; image-size : 1024x1024 <br>
&emsp; number of images : 1100 <br>
&emsp; epoch : 2

