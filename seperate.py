import os
from shutil import copy

file_path1 = '/home/prak12-2/CrossScene/cropped_images'
file_path2 = '/home/prak12-2/CrossScene/density_maps'
sav_dir_val_1 = '/home/prak12-2/CrossScene/valid_images'
sav_dir_val_2 = '/home/prak12-2/CrossScene/valid_density'
sav_dir_train_1 = '/home/prak12-2/CrossScene/train_images'
sav_dir_train_2 = '/home/prak12-2/CrossScene/train_density'

file_img = os.listdir(file_path1)
file_den = os.listdir(file_path2)


for i in range(14400):

    from_path_train_1 = os.path.join(file_path1,file_img[i])
    to_path_train_1 = os.path.join(sav_dir_train_1,file_img[i])
    copy(from_path_train_1, to_path_train_1)

    from_path_train_2 = os.path.join(file_path2,file_den[i])
    to_path_train_2 = os.path.join(sav_dir_train_2,file_den[i])
    copy(from_path_train_2, to_path_train_2)

for i in range(14401,18000):
    from_path_valid_1 = os.path.join(file_path1,file_img[i])
    to_path_valid_1 = os.path.join(sav_dir_val_1,file_img[i])
    copy(from_path_valid_1, to_path_valid_1)

    from_path_valid_2 = os.path.join(file_path2,file_den[i])
    to_path_valid_2 = os.path.join(sav_dir_val_2,file_den[i])
    copy(from_path_valid_2, to_path_valid_2)


print("the processing is done")


