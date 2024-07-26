"""Code to rename image files"""
import os


def files_rename(folder_path):
    """FUNCTION TO RENAME FILES"""
    files = os.listdir(folder_path)


   # Rename the first 45 files
    for i in range(45):
        old_file = os.path.join(folder_path, files[i])
        new_file = os.path.join(folder_path, f"Image_2024071507{str(i).zfill(2)}.jpg")
        os.rename(old_file, new_file)

   # Rename the next 75 files
    for i in range(75):
        old_file = os.path.join(folder_path, files[45 + i])
        if i < 60:
            new_file = os.path.join(folder_path,f"Image_2024071508{str(i).zfill(2)}.jpg")
        else:
            new_file = os.path.join(folder_path,f"Image1_2024071508{str(60-i).zfill(2)}.jpg")
        os.rename(old_file, new_file)

   # Rename the next 55 files
    for i in range(55):
        old_file = os.path.join(folder_path, files[120 + i])
        new_file = os.path.join(folder_path, f"Image_2024071509{str(i).zfill(2)}.jpg")
        os.rename(old_file, new_file)

   # Rename the last 57 files
    for i in range(57):
        old_file = os.path.join(folder_path, files[175 + i])
        new_file = os.path.join(folder_path, f"Image_2024071510{str(i).zfill(2)}.jpg")
        os.rename(old_file, new_file)


files_rename("TEST")
