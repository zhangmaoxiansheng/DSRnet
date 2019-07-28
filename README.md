# RGB guided depth map super-resolution #


## Environment ##
Linux

pytorch 0.4.1(python 2.7)

tensorboardX and tensorflow(just for visualization)
         
      
## Usage ##

Your dataset should include thress sub-directory: lr* hr* img*

Warning: You should use the same format to name your images and PFM files

For training:

```
python main.py --maxdisp 192 --model basic --datapath ./dataset -- epoches 50 --learning 1e-5 --savemodel ./trained_model --log_dir ./log
```

For testing

one image

```
python simple_test.py --img ./dataset/img_20190503_4/0m.png --lr ./dataset/lr_disp/0_disp.pfm --output_dir ./result --loadmodel ./trained_model/checkpoint_40.tar
```
or put your images and lr_pfms in your folders

```
python simple_test.py --img ./dataset/img_20190503_4 --lr ./dataset/lr_disp --output_dir ./result --loadmodel ./trained_model/checkpoint_40.tar
```

output: imagebasename_sr_vis.png and imagebasename_sr.pfm












