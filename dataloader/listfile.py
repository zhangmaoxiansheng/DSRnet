import torch.utils.data as data

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataloader(root_path):

    paths = [os.path.join(root_path, d) for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]
    img_path_ = [d for d in paths if d.find('img') > -1]
    img_path = img_path_[0]
    lr_path_ = [d for d in paths if d.find('lr') > -1]
    lr_path = lr_path_[0]
    hr_path_ = [d for d in paths if d.find('hr') > -1]
    hr_path = hr_path_[0]

    img_file = os.listdir(img_path)
    img_file.sort()
    #print(img_file)
    img_file = [os.path.join(img_path,d) for d in img_file]
    all_img = img_file
    test_img = img_file[1:50]

    lr_file = os.listdir(lr_path)
    lr_file.sort()
    #print(lr_file)
    lr_file = [os.path.join(lr_path,d) for d in lr_file]
    all_lr = lr_file
    test_lr = all_lr[1:50]
    
    hr_file = os.listdir(hr_path)
    hr_file.sort()
    #print(hr_file)
    hr_file = [os.path.join(hr_path,d) for d in hr_file]
    all_hr = hr_file
    test_hr = all_hr[1:50]

    return all_img, all_lr, all_hr, test_img, test_lr, test_hr