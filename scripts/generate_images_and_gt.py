import argparse 
import h5py
import os
import cv2
import numpy as np

def generate_images_and_gt_from_hdf(args):
    file_path  = args.file_path
    dataset_folder_path = args.dataset_folder_path

    _, hdf_file_name = os.path.split(file_path)

    with h5py.File(file_path, "r") as f:
        # List all groups
        print("Keys: %s" % f.keys())

        # Get the data
        volume=f['volumes']['raw'][()]
        labels=f['volumes']['labels']['clefts'][()]

    print(volume.shape)
    print(labels.shape)

    # generate_image(os.path.join(dataset_folder_path, hdf_file_name, "img"), volume, hdf_file_name)
    generate_gt(os.path.join(dataset_folder_path, hdf_file_name, "gt"), labels, hdf_file_name)


def generate_image(folder_path, volume, hdf_file_name):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        for i in range(0, volume.shape[0]):
            img = volume[i, :, :]
            print(img.shape)
            img_filename = os.path.join(folder_path, hdf_file_name+f"_{i}.jpg") 
            cv2.imwrite(img_filename, img)
    else:
        print(f"directory {folder_path} exists")
            

def generate_gt(folder_path, labels, hdf_file_name, false_value = 18446744073709551615):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        for i in range(0, labels.shape[0]):
            gt_img = labels[i, :, :]
            gt_img[gt_img==false_value] = 0
            gt_img = gt_img.astype(bool) 
            gt_img = np.array(gt_img, dtype=np.uint8)*255   
            gt_img_filename = os.path.join(folder_path, hdf_file_name+f"_{i}.jpg") 
            cv2.imwrite(gt_img_filename, gt_img)
    else:
        print(f"directory {folder_path} exists")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', help='path to the hdf file')
    parser.add_argument('--dataset_folder_path', help='path to store the data_set')
    args = parser.parse_args()

    generate_images_and_gt_from_hdf(args)