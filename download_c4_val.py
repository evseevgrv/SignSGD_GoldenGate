####


import os
import requests
import tarfile
import shutil

BASE_URL = "https://huggingface.co/datasets/allenai/c4/resolve/main/en/c4-validation.{:05d}-of-00008.json.gz"
START_INDEX = 0
END_INDEX = 8
BATCH_SIZE = 1
DOWNLOAD_DIR = "downloads"
ARCHIVE_DIR = "archives"

def download_file(url, dest_folder):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        filename = url.split('/')[-1]
        file_path = os.path.join(dest_folder, filename)
        with open(file_path, 'wb') as file:
            shutil.copyfileobj(response.raw, file)
        print(f"Downloaded: {filename}")
    else:
        print(f"Failed to download: {url}")

def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def archive_files(folder_path, archive_name):
    with tarfile.open(archive_name, 'w:gz') as tar:
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            tar.add(file_path, arcname=file_name)
    print(f"Archived: {archive_name}")

def delete_folder(folder_path):
    shutil.rmtree(folder_path)
    print(f"Deleted folder: {folder_path}")

def main():
    # create_folder(DOWNLOAD_DIR)
    # create_folder(ARCHIVE_DIR)

    for start in range(START_INDEX, END_INDEX + 1, BATCH_SIZE):
        # end = min(start + BATCH_SIZE - 1, END_INDEX)
        # batch_folder = os.path.join(DOWNLOAD_DIR, f"batch_{start:05d}_{end:05d}")
        # create_folder(batch_folder)
        download_dir = os.path.join(os.getcwd(), "c4_val")
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)
        for i in range(start, start + BATCH_SIZE):
            if i > END_INDEX:
                break
            url = BASE_URL.format(i)
            download_file(url, download_dir)
        
        # archive_name = os.path.join(ARCHIVE_DIR, f"batch_{start:05d}_{end:05d}.tar.gz")
        # archive_files(batch_folder, archive_name)
        # delete_folder(batch_folder)

main()