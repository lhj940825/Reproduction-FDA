# Created by Hojun Lim(Media Informatics, 405165) at 28.08.20

import argparse
import json
import os
from shutil import copyfile

class BDD100k_preprocessor():
    """
    Class for preprocessing(sorting out the images by its weather) the BDD100K dataset
    Sort the images by its weather and store them(with corresponding labels) in different folders by weather
    """

    def __init__(self, args: argparse):
        self.BDD100k_json_path=args.C_Driving_json_path
        self.BDD100k_json_file_name = args.C_Driving_json_file_name
        self.BDD100k_dir = args.C_Driving_dir
        self.BDD100k_image_dir = args.C_Driving_source_image_dir
        self.BDD100k_label_dir = args.C_Driving_target_image_label_dir
        self.root=args.root

        self.weathers = "rainy|snowy|clear|overcast|undefined|partly cloudy|foggy"



    def split_dataset(self):
        # extract name and weather of each image from BDD100k dataset
        self.BDD100k_data = [{'id': i_data['name'], 'weather': i_data['attributes']['weather']} for i_data in
                             json.load(open(os.path.join(self.BDD100k_json_path, self.BDD100k_json_file_name)))]

        # check whether this json file is for training set or validation set
        if 'val' in self.BDD100k_json_file_name:
            data_type = 'val'
        elif 'train' in self.BDD100k_json_file_name:
            data_type = 'train'
        else:
            raise ValueError()

        # directories where the images and corresponding labels lie in BDD100k dataset
        origin_image_dir = os.path.join(self.BDD100k_dir, self.BDD100k_image_dir, data_type)
        origin_label_dir = os.path.join(self.BDD100k_dir, self.BDD100k_label_dir, data_type)

        # make directories needed to store sorted images and labels and retrieve the directories where the sorted images and labels will be stored
        _destin_image_dir, _destin_label_dir = self.build_directory_structure()

        # Note that out of 60000(approximate) 'id' of images in bdd100k json file, only 3000 images actually exist in bdd100k dataset.
        image_not_found_cnt = 0
        total_image_cnt = 0
        found_image_list = [] # list to store the id and weather attributes of found images(co-existing in bdd100k json file and bdd100k dataset.

        for i_data in self.BDD100k_data:
            total_image_cnt += 1

            destin_image_dir = os.path.join(_destin_image_dir, i_data['weather'], data_type)
            destin_label_dir = os.path.join(_destin_label_dir, i_data['weather'], data_type)

            # copy the images and labels from source dir to target dir
            try:
                copyfile(os.path.join(origin_image_dir, i_data['id']), os.path.join(destin_image_dir, i_data['id']))
                copyfile(os.path.join(origin_label_dir, i_data['id'].replace('.jpg', '_train_id.png')),
                         os.path.join(destin_label_dir, i_data['id'].replace('.jpg', '_train_id.png')))

                found_image_list.append(i_data)

            except Exception:
                image_not_found_cnt +=1

        print('{}-image found from {} containing {} images'.format( total_image_cnt - image_not_found_cnt, self.BDD100k_json_file_name, total_image_cnt))


        with open(os.path.join(self.BDD100k_json_path, str(data_type)+'.txt'), 'w') as file:
            for image_id in found_image_list:
                if image_id['weather'] == 'clear': # currently we only use images whose weather attribute is 'clear'
                    file.write(image_id['id'] + '\n')



    def build_directory_structure(self):
        """
        Example of produced structure is as follows

        bdd100k
        -images
        ---sunny
        -----train
        -----test
        -----val
        ---rainy
        -----train
        -----test
        -----val

        .
        .
        .

        -labels
        ---sunny
        -----train
        -----test
        -----val


        """
        self.weathers = self.weathers.split('|')

        image_dir = os.path.join(self.root, 'images')
        os.makedirs(image_dir, exist_ok=True)  # make a folder to store images(sorted, classified by weather)
        label_dir = os.path.join(self.root, 'labels')
        os.makedirs(label_dir, exist_ok=True)  # make a folder to store corresponding gt labels

        sub_folders = ['test', 'train', 'val']
        for weather in self.weathers:

            _dir = os.path.join(image_dir, weather)
            os.makedirs(_dir, exist_ok=True)
            for sub_folder in sub_folders:
                os.makedirs(os.path.join(_dir, sub_folder), exist_ok=True)

            _dir = os.path.join(label_dir, weather)
            os.makedirs(_dir, exist_ok=True)
            for sub_folder in sub_folders:
                os.makedirs(os.path.join(_dir, sub_folder), exist_ok=True)

        return image_dir, label_dir


if __name__ == '__main__':
    #TODO Before run this, place the bdd100k_labels_images_train.json and bdd100k_labels_images_val.json under the ./datset/BDD100k_list
    # also place the BDD100k dataset under ../data_semseg

    parser = argparse.ArgumentParser(description="script for preprocessing the BDD100K dataset: Sort out the images by weather")
    parser.add_argument('--BDD100k_json_path', type=str, default='./dataset/BDD100k_list')
    parser.add_argument('--BDD100k_json_file_name', type=str, default='bdd100k_labels_images_train.json')
    parser.add_argument('--BDD100k_dir', type=str, default='../data_semseg/bdd100k', help='where the actual dataset(bdd100k folder) is')
    parser.add_argument('--BDD100k_image_dir', type=str, default='seg/images')
    parser.add_argument('--BDD100k_label_dir', type=str, default='seg/labels')
    parser.add_argument('--root', type=str, default='../data_semseg/bdd100k')
    #parser.add_argument('--sorted_dataset_dir', type=str, default='')

    args = parser.parse_args()

    bdd100k_preprocessor = BDD100k_preprocessor(args)
    bdd100k_preprocessor.split_dataset()

# command(at /media/data/hlim/FDA/FDA)
# ##  python3 utils/BDD100k_preprocessor.py