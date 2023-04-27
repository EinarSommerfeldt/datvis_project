import os

# dataset root dir
path = "C:/Users/einar/Desktop/datvis_project/RDD2022_dataset/RDD2022_released_through_CRDDC2022/RDD2022/"
path = "/cluster/projects/itea_lille-idi-tdt4265/datasets/rdd2022/RDD2022/" #idun
path = "/content/RDD2022/" #colab

dirs = ["China_Drone/train/images", 
	"China_MotorBike/train/images", 
	"Czech/train/images", 
	"India/train/images", 
	"Japan/train/images", 
	"Norway/train/images", 
	"United_States/train/images"] 

def fill_image_paths(train_filename, val_filename, train_prop: float):
    train_file = open(train_filename, "w")
    val_file = open(val_filename, "w")

    for dir in dirs:
        dir_path = path + dir
        file_list = os.listdir(dir_path)
        for i, filename in enumerate(file_list):
            line = dir_path+os.sep+filename+'\n'
            if i < train_prop*len(file_list):
                train_file.write(line.replace("/", os.sep))
            else:
                val_file.write(line.replace("/", os.sep))


#idun
#fill_image_paths("/cluster/home/einarjso/datvis_project/yolo/data_info/train.txt","/cluster/home/einarjso/datvis_project/yolo/data_info/val.txt", 0.9)
#colab
fill_image_paths("/content/train.txt","/content/val.txt", 0.9)