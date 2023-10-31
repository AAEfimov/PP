# import package
import labelme2coco

# set directory that contains labelme annotations and image files
labelme_folder = "stones"

# set export dir
export_dir = "stones"

# set train split rate
#train_split_rate = 0.85

# convert labelme annotations to coco
labelme2coco.convert(labelme_folder, export_dir)

