import os

dataset_path = r"C:\Users\prave\OneDrive\Pictures\Desktop\CASIA2"

extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

folders = ["Au", "Tp", "CASIA 2 Groundtruth"]

for folder in folders:

    folder_path = os.path.join(dataset_path, folder)

    if not os.path.exists(folder_path):
        print(f"{folder}: Folder not found!")
        continue

    count = 0

    for root, dirs, files in os.walk(folder_path):
        count += sum(
            file.lower().endswith(extensions)
            for file in files
        )

    print(f"{folder}: {count} images")

