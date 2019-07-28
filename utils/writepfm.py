import re
import numpy as np
import sys
def write_pfm(file, image, scale=1):
    """ Write a pfm file """
    file = open(file, 'wb')

    color = None

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    if len(image.shape) == 3 and image.shape[2] == 3: # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # grayscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n'.encode('utf-8')  if color else 'Pf\n'.encode('utf-8'))
    temp_str = '%d %d\n' % (image.shape[1], image.shape[0])
    file.write(temp_str.encode('utf-8'))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    temp_str = '%f\n' % scale
    file.write(temp_str.encode('utf-8'))

    image.tofile(file)