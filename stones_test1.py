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

val_dict = get_stone_dict(traint_dataset_val_pwd)

for d in random.sample(val_dict, 2):
    img   = Image.open(d["file_name"])
    _,axs = plt.subplots(1,2,figsize=(12,8))
    axs[0].imshow(img); axs[0].axis('off')
    visualizer = Visualizer(img, metadata=stone_metadata, scale=1)
    out = visualizer.draw_dataset_dict(d)
    axs[1].imshow(out.get_image()); axs[1].axis('off')
    plt.tight_layout();plt.show()


