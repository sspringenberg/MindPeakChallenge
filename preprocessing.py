import os
import argparse
from PIL import Image

parser = argparse.ArgumentParser(description='MPChallenge Preproessing')
parser.add_argument('--source-root', type=str, metavar='S',
                    help='original data directory')
parser.add_argument('--dest-root', type=str, metavar='S',
                    help='destination directory')
parser.add_argument('--width-height', type=int, metavar='N',
                    help='max width / height')

def preprocess_images(source_root, dest_root, width_height):
    """
    preprocess imges in --source_root by reducing size, save prepocessed
    images to --dest_root
    """

    os.mkdir(os.path.join(dest_root))
    for cl in os.listdir(source_root):
        os.mkdir(os.path.join(dest_root, cl))
        cl_folder = os.path.join(source_root, cl)

        for curr_image in os.listdir(cl_folder):
            im = Image.open(os.path.expanduser(os.path.join(cl_folder, curr_image)))
            im.thumbnail((width_height, width_height))
            im.save(os.path.splitext(os.path.join(dest_root, cl, curr_image))[0]+".png", "PNG")

if __name__ == "__main__":
    args = parser.parse_args()
    if (args.source_root or args.dest_root) is None :
        raise TypeError("Source folder and / or destination folder"\
        "not passed as argument (see -h)")
    preprocess_images(args.source_root, args.dest_root, args.width_height)
