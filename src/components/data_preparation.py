import os
import sys
import torch
import numpy as np
from PIL import Image
from src.logger import logging
from src.exception import CustomException


class PennFudanDataset(torch.utils.data.Dataset):

    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        try:
            logging.info("load images and masks")
            img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
            mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
            img = Image.open(img_path).convert("RGB")
            logging.info("Note: that we haven't converted the mask to RGB, "
                         "because each color corresponds to a different instance with 0 being background")
            mask = Image.open(mask_path)

            mask = np.array(mask)
            logging.info("Instances are encoded as different colors")
            obj_ids = np.unique(mask)
            logging.info("First id is the background, so remove it")
            obj_ids = obj_ids[1:]

            logging.info("Split the color-encoded mask into a set of binary masks")
            masks = mask == obj_ids[:, None, None]

            logging.info("Get bounding box coordinates for each mask")
            num_objs = len(obj_ids)
            boxes = []
            for i in range(num_objs):
                pos = np.where(masks[i])
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                boxes.append([xmin, ymin, xmax, ymax])

            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            logging.info("There is only one class")
            labels = torch.ones((num_objs,), dtype=torch.int64)
            masks = torch.as_tensor(masks, dtype=torch.uint8)

            image_id = torch.tensor([idx])
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            logging.info("Suppose all instances are not crowd")
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            target["masks"] = masks
            target["image_id"] = image_id
            target["area"] = area
            target["iscrowd"] = iscrowd

            if self.transforms is not None:
                img, target = self.transforms(img, target)

            return img, target
        except Exception as e:
            raise CustomException(e, sys) from e

    def __len__(self):
        try:
            return len(self.imgs)
        except Exception as e:
            raise CustomException(e, sys) from e
