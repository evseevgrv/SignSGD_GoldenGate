import os
import re
import glob
from shutil import move
from torchvision.datasets.utils import download_and_extract_archive, extract_archive

def prepare_imagenet_val(data_root='./data/imagenet-1k'):
    """
    Собирает папку data_root/val из файлов вида ILSVRC2012_val_00050000.JPEG.
    Если нет wnids.txt — берёт названия классов из папки train/.
    Если нет validation_ground_truth.txt — пробует скачать devkit.
    """
    # пути
    train_dir = os.path.join(data_root, 'train')
    val_folder = os.path.join(data_root, 'val')
    devkit_dir = os.path.join(data_root, 'ILSVRC2012_devkit_t12')
    archive_base = os.path.dirname(data_root)
    devkit_tar = os.path.join(archive_base, 'ILSVRC2012_devkit_t12.tar.gz')

    # 1) пытаемся найти файл аннотаций
    ann_list = glob.glob(os.path.join(devkit_dir, '**', 'ILSVRC2012_validation_ground_truth.txt'),
                         recursive=True)
    if not ann_list:
        # скачиваем и распаковываем devkit
        print("📦 Скачиваем/перезапаковываем devkit для аннотаций...")
        if os.path.isdir(devkit_dir):  # старый мусор — удаляем
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
        if not ann_list:
            raise RuntimeError("Не найден файл validation_ground_truth.txt даже после скачивания devkit")
    ann_path = ann_list[0]
    gt = [int(x) for x in open(ann_path, 'r')]

    # 2) получаем список классов (wnids)
    wnids_list = glob.glob(os.path.join(devkit_dir, '**', 'wnids.txt'),
                           recursive=True)
    if wnids_list:
        wnids = [x.strip() for x in open(wnids_list[0], 'r')]
    else:
        # fallback: берём папки из train/
        print("⚠️ wnids.txt не найден — формируем список классов из папки train/")
        if not os.path.isdir(train_dir):
            raise RuntimeError("Папка train/ отсутствует, а wnids.txt не найден")
        wnids = sorted(d for d in os.listdir(train_dir)
                       if os.path.isdir(os.path.join(train_dir, d)))

    # 3) создаём папку val и раскладываем файлы
    os.makedirs(val_folder, exist_ok=True)
    pattern = os.path.join(data_root, 'ILSVRC2012_val_*.JPEG')
    files = glob.glob(pattern)
    if not files:
        raise RuntimeError("Не найдены файлы ILSVRC2012_val_*.JPEG в " + data_root)

    for img_path in files:
        img_name = os.path.basename(img_path)
        idx = int(re.search(r'_(\d+)\.JPEG', img_name, re.IGNORECASE).group(1))
        cls = wnids[gt[idx-1] - 1]
        dst_dir = os.path.join(val_folder, cls)
        os.makedirs(dst_dir, exist_ok=True)
        move(img_path, os.path.join(dst_dir, img_name))

    total = sum(len(files) for _,_,files in os.walk(val_folder))
    print(f"✅ Готово: {len(wnids)} классов, {total} изображений в {val_folder}")

if __name__ == '__main__':
    prepare_imagenet_val()
