import torch
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION  = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)

import os

import detectron2
import torch, torchvision

import numpy as np
import os, json, cv2, random
from PIL import Image
import matplotlib.pyplot as plt

from detectron2        import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data             import MetadataCatalog, DatasetCatalog
from detectron2.structures       import BoxMode
from detectron2.data.datasets    import register_coco_instances, load_coco_json
from detectron2.utils.logger     import setup_logger
from detectron2.utils.visualizer import ColorMode
from detectron2.evaluation       import COCOEvaluator, inference_on_dataset
from detectron2.data             import build_detection_test_loader

import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

import os
import copy
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import transforms, datasets
import torchvision.transforms as T

from torchsummary import summary

from tqdm.notebook import tqdm, trange

import pandas as pd


# from albumentations.pytorch import ToTensorV2
# import albumentations as A

import shutil
from urllib.request import urlretrieve
from torchvision.datasets.utils import download_and_extract_archive


from detectron2.model_zoo.model_zoo import _ModelZooUrls
mz = _ModelZooUrls()
print("\n".join(list(mz.CONFIG_PATH_TO_URL_SUFFIX.keys())))


dataset_directory = "./"
content = os.listdir(dataset_directory)
print(content)


dataset_catalog_name = "stone_train"
dataset_catalog_name_val = "stone_val"

try:
    DatasetCatalog.pop(dataset_catalog_name)
except:
    pass

def get_stone_dict(fp):
  fn = fp + "dataset.json"
  with open(fn) as f:
    data = json.load(f)

  dataset_dicts = []

  for img in data['images']:
    record = {}

    filename = os.path.basename(img["file_name"]).replace(":", "_")
    filename = os.path.join(fp, filename)

    print(filename)

    record["file_name"] = filename
    record["image_id"] = img['id']
    record["height"] = img['height']
    record["width"] = img['width']

    objs = []
    for anno in data['annotations']:
      if anno['image_id'] == record["image_id"]:
        obj = {
            "bbox" : anno['bbox'],
            # "bbox_mode": BoxMode.XYWH_ABS,
            "bbox_mode" : BoxMode.XYXY_ABS,
            "segmentation": anno['segmentation'],
            "category_id": 0,
            # anno['category_id'],
            "iscrowd": 0,

        }

        objs.append(obj)

    record["annotations"] = objs
    dataset_dicts.append(record)

  return dataset_dicts

# traint_dataset_pwd = dataset_directory + "/train/"
# print(get_stone_dict(traint_dataset_pwd))


traint_dataset_pwd = dataset_directory + "/train/"
traint_dataset_val_pwd = dataset_directory + "/val/"


try:
    DatasetCatalog.register(dataset_catalog_name, lambda : get_stone_dict(traint_dataset_pwd))
except:
    print(f'Probably data {traint_dataset_pwd} have been already registred')

try:
    DatasetCatalog.register(dataset_catalog_name_val, lambda : get_stone_dict(traint_dataset_val_pwd))
except:
    print(f'Probably data {traint_dataset_val_pwd} have been already registred')

MetadataCatalog.get(dataset_catalog_name).set(thing_classes=["stone"])
MetadataCatalog.get(dataset_catalog_name_val).set(thing_classes=["stone"])

stone_metadata = MetadataCatalog.get("stone_train")
dataset_dicts    = get_stone_dict(traint_dataset_pwd)

print(stone_metadata.as_dict())

print('N Train',len(dataset_dicts))
print(dataset_dicts)

#val_dict = get_stone_dict(traint_dataset_val_pwd)
"""
for d in random.sample(val_dict, 2):
    img   = Image.open(d["file_name"])
    _,axs = plt.subplots(1,2,figsize=(12,8))
    axs[0].imshow(img); axs[0].axis('off')
    visualizer = Visualizer(img, metadata=stone_metadata, scale=1)
    out = visualizer.draw_dataset_dict(d)
    axs[1].imshow(out.get_image()); axs[1].axis('off')
    plt.tight_layout();plt.show()
"""

from detectron2.engine import DefaultTrainer
from detectron2.data import build_detection_test_loader, build_detection_train_loader

class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=custom_mapper)


from detectron2.data import detection_utils as utils
import detectron2.data.transforms as T
import copy
import torch

def custom_mapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

    image = utils.read_image(dataset_dict["file_name"], format="BGR")

    transform_list = [
                        T.Resize((800,600)),
                        T.RandomBrightness(0.8, 1.8),
                        T.RandomContrast(0.6, 1.3),
                        T.RandomSaturation(0.8, 1.4),
                        T.RandomRotation(angle=[90, 90]),
                        T.RandomLighting(0.7),
                        T.RandomFlip(prob=0.4, horizontal=False, vertical=True),
                    ]

    image, transforms = T.apply_transform_gens(transform_list, image)

    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

    annos = [
        utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop("annotations")
        if obj.get("iscrowd", 0) == 0
    ]

    instances = utils.annotations_to_instances(annos, image.shape[:2])

    dataset_dict["instances"] = utils.filter_empty_instances(instances)

    return dataset_dict


BATCH = 2

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("stone_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = BATCH
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)

import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True);
trainer = CustomTrainer(cfg)#DefaultTrainer(cfg);

trainer.resume_or_load(resume=False)

# TRAIN!!!!!
trainer.train()


cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
cfg.DATASETS.TEST = ("stone_val", )
predictor = DefaultPredictor(cfg)

from detectron2.utils.visualizer import ColorMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader


dataset_dicts_val = get_stone_dict(traint_dataset_val_pwd)
print(dataset_dicts_val)

for d in random.sample(dataset_dicts_val, 2):
    img   = Image.open(d["file_name"])
    _,axs = plt.subplots(1,2,figsize=(12,8))
    axs[0].imshow(img); axs[0].axis('off')
    visualizer = Visualizer(img, metadata=stone_metadata, scale=1)
    out = visualizer.draw_dataset_dict(d)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    axs[i].imshow(out.get_image()); axs[i].axis('off');

plt.show()

