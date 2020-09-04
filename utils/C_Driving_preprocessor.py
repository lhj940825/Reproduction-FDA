# Created by Hojun Lim(Media Informatics, 405165) at 31.08.20

import argparse
import os

class C_Driving_preprocessor():
    """
    Class for preprocessing(sorting out the images by its weather) the C-Driving dataset

    """

    def __init__(self, args: argparse):
        #self.C_Driving_json_path=args.C_Driving_json_path
        #self.C_Driving_json_file_name = args.C_Driving_json_file_name
        self.C_Driving_dir = args.root
        self.weathers = args.weathers.split('|')
        self.C_Driving_source_image_dir = [os.path.join(self.C_Driving_dir, 'train', 'compound', weather) for weather in self.weathers[:3]] # for source dataset, there is no corresponding label
        self.C_Driving_source_image_dir.append(os.path.join(self.C_Driving_dir, 'train', 'open_not_used', self.weathers[3])) # since the weather[3] = 'overcast' has different path
        self.C_Driving_target_image_label_dir = [os.path.join(self.C_Driving_dir, 'val', 'compound', weather) for weather in self.weathers[:3]]
        self.C_Driving_target_image_label_dir.append(os.path.join(self.C_Driving_dir, 'val', 'open',  self.weathers[3])) # since the weather[3] = 'overcast' has different path
        self.C_Driving_list_path = args.C_Driving_list_path


    def preprocess(self):
        for source_dir, target_dir, weather in zip(self.C_Driving_source_image_dir, self.C_Driving_target_image_label_dir, self.weathers):

            # make a txt file that stores all source image file names in it
            self.wrtie_txt(source_dir, weather, self.C_Driving_list_path)
            self.wrtie_txt(target_dir, weather, self.C_Driving_list_path)

    def wrtie_txt(self, dataset_dir,  weather, list_path):
        """

        list_path: where to store a txt file containing list of all image id in the given dataset_dir
        """

        data_type = 'train' if 'train' in dataset_dir else 'val'
        with open(os.path.join(list_path, '%s_%s.txt' %(data_type, weather)), 'w') as file:
            for image_id in os.listdir(dataset_dir):
                if 'train_id' not in image_id: # only write the id of images. if image_id contains 'train_id' then such files are label images and don't write them into the txt file.
                    file.write(image_id + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="script for preprocessing the C_Driving dataset: Sort out the images by weather")
    parser.add_argument('--C_Driving_list_path', type=str, default='./dataset/C_Driving_list')
    parser.add_argument('--root', type=str, default='../data_semseg/C-Driving', help='where the actual dataset(C_Driving folder) is')
    parser.add_argument('--weathers', type=str, default="cloudy|rainy|snowy|overcast")
    args = parser.parse_args()

    c_driving_preprocessor = C_Driving_preprocessor(args)
    c_driving_preprocessor.preprocess()

# command(at /media/data/hlim/FDA/FDA)
# ##  python3 utils/BDD100k_preprocessor.py