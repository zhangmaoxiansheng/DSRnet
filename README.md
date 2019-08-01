# RGB guided depth map super-resolution #


## Environment ##
Linux

pytorch 0.4.1(python 2.7)

tensorboardX and tensorflow(just for visualization)
         
      
## Usage ##

Your dataset should include thress sub-directory: lr* hr* img*

Warning: You should use the same format to name your images and PFM files

For training basic_res:

```
python main.py --maxdisp 144 --model basic_res --mask_disp 0.25 --datapath ./dataset -- epoches 50 --learning 1e-4 --savemodel ./trained_model --log_dir ./log
```

Decrease the mask_disp each time can improve the proformance. For example, the first time, mask_disp = 0.25, the second time, mask_disp = 0.15, the third time, mask_disp = 0

For training iternativenet:

The first stage:

```
python main.py --maxdisp 144 --model itnet --stage first --mask_disp 0.25 --datapath ./dataset -- epoches 50 --learning 1e-4 --savemodel ./trained_model --log_dir ./log
```

The second stage(distill stage):

```
python main.py --maxdisp 144 --model itnet --stage distill --loadmodel ./trained_model/checkpoint_50.tar --mask_disp 0 --datapath ./dataset -- epoches 50 --learning 1e-4 --savemodel ./trained_model --log_dir ./log
```

For testing

one image

```
python simple_test.py  --model basic_res--img ./dataset/img_20190503_4/0m.png --lr ./dataset/lr_disp/0_disp.pfm --output_dir ./result --loadmodel ./trained_model/checkpoint_40.tar
```
or put your images and lr_pfms in your folders

```
python simple_test.py --model basic_res --img ./dataset/img_20190503_4 --lr ./dataset/lr_disp --output_dir ./result --loadmodel ./trained_model/checkpoint_40.tar
```

output: imagebasename_sr_vis.png, imagebasename_sr.pfm and imagebasename_res.pfm

If you use itnet, you should match the stage with the checkpoint. The output of the first stage is just like basic_res, and the output of the distill stage include imagebasename_res1.pfm imagebasename_res2.pfm in addition to basic output.












