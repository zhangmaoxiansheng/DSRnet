from readpfm import readPFM
import matplotlib.pyplot as plt
import os
import argparse

parser = argparse.ArgumentParser(description='pfm visualization')

parser.add_argument('--input_dir', type=str, default='./dataset/lr_disp',
                    help='depth_path')
parser.add_argument('--output_dir', type=str, default='./result',
                    help='outputdir')
args = parser.parse_args()

if os.path.isdir(args.input_dir):
    test_pfm = os.listdir(args.input_dir)
    test_pfm.sort()
    pfm_path = [os.path.join(args.input_dir,d) for d in test_pfm]
else:
    pfm_path = [args.input_dir]
print("there are %d files"%(len(pfm_path)))
basenames = [os.path.basename(f) for f in pfm_path]
basenames = [os.path.splitext(f)[0] for f in basenames]

for num,path in enumerate(pfm_path):
    output = readPFM(path)
    plt.imsave(str(os.path.join(args.output_dir,basenames[num]) + "_vis.png"), output, cmap = 'plasma')
    print("finished %d"%(num+1))
