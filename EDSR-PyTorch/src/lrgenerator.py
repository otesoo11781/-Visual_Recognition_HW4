import glob
import cv2
import os
import argparse

parser = argparse.ArgumentParser(description='low resolution image generator.')

parser.add_argument('--dir_hr', type=str, default='../../datasets/training_hr_images',
                    help='high resolution images directory')
parser.add_argument('--dir_lr', type=str, default='../../datasets/training_lr_images',
                    help='low resolution images directory')
parser.add_argument('--scale', type=int, default=3,
                    help='down scale factor')
parser.add_argument('--ext', type=str, default='.png',
                    help='image file extension')


def create_lr_images(dir_hr, dir_lr, s, ext):
    assert os.path.exists(dir_hr), 'Cannot find high resolution image path.'

    if not os.path.exists(dir_lr):
        os.mkdir(dir_lr)

    subdir_lr = os.path.join(dir_lr, 'X{}'.format(s))
    if not os.path.exists(subdir_lr):
        os.mkdir(subdir_lr)

    names_hr = sorted(
        glob.glob(os.path.join(dir_hr, '*' + ext))
    )
    assert len(names_hr) > 0, 'Cannot find any high resolution images.'
    for f in names_hr:
        filename, _ = os.path.splitext(os.path.basename(f))
        hr_img = cv2.imread(f)
        hr = hr_img.shape  # (h, w, 3)

        lr = (int(hr[1]/s), int(hr[0]/3)) # (w, h)
        lr_img = cv2.resize(hr_img, lr, interpolation=cv2.INTER_CUBIC)  # bicubic down-sample
        cv2.imwrite(os.path.join(
            dir_lr, 'X{}/{}x{}{}'.format(
                s, filename, s, ext
            )
        ), lr_img)
    print('Create low resolution images done!')


args = parser.parse_args()
create_lr_images(args.dir_hr, args.dir_lr, args.scale, args.ext)
