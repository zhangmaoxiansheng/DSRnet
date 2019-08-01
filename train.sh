python main.py --learning_rate 1e-4 --savemodel ./trained_model_dilate
python main.py --learning_rate 1e-4 --mask_disp 0.15 --loadmodel ./trained_model_dilate/checkpoint_100.tar --savemodel ./trained_model_dilate2
