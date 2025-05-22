import os
import re
import glob
from shutil import move
from torchvision.datasets.utils import download_and_extract_archive, extract_archive

def prepare_imagenet_val(data_root='./data/imagenet-1k'):
    train_dir = os.path.join(data_root, 'train')
    val_folder = os.path.join(data_root, 'val')
    devkit_dir = os.path.join(data_root, 'ILSVRC2012_devkit_t12')
    archive_base = os.path.dirname(data_root)
    devkit_tar = os.path.join(archive_base, 'ILSVRC2012_devkit_t12.tar.gz')

    ann_list = glob.glob(os.path.join(devkit_dir, '**', 'ILSVRC2012_validation_ground_truth.txt'),
                         recursive=True)
    if not ann_list:
        if os.path.isdir(devkit_dir): 
            import shutil
            shutil.rmtree(devkit_dir)
        download_and_extract_archive(
            url='https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz',
            download_root=archive_base,
            extract_root=data_root,
            filename=os.path.basename(devkit_tar)
        )
        ann_list = glob.glob(os.path.join(devkit_dir, '**', 'ILSVRC2012_validation_ground_truth.txt'),
                             recursive=True)
    ann_path = ann_list[0]
    gt = [int(x) for x in open(ann_path, 'r')]

    wnids_list = glob.glob(os.path.join(devkit_dir, '**', 'wnids.txt'),
                           recursive=True)
    if wnids_list:
        wnids = [x.strip() for x in open(wnids_list[0], 'r')]
    else:
        wnids = sorted(d for d in os.listdir(train_dir)
                       if os.path.isdir(os.path.join(train_dir, d)))

    os.makedirs(val_folder, exist_ok=True)
    pattern = os.path.join(data_root, 'ILSVRC2012_val_*.JPEG')
    files = glob.glob(pattern)

    for img_path in files:
        img_name = os.path.basename(img_path)
        idx = int(re.search(r'_(\d+)\.JPEG', img_name, re.IGNORECASE).group(1))
        cls = wnids[gt[idx-1] - 1]
        dst_dir = os.path.join(val_folder, cls)
        os.makedirs(dst_dir, exist_ok=True)
        move(img_path, os.path.join(dst_dir, img_name))

    total = sum(len(files) for _,_,files in os.walk(val_folder))

if __name__ == '__main__':
    prepare_imagenet_val()
