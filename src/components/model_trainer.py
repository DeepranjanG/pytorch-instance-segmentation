import os
import sys
import torch
import torchvision
from src.logger import logging
from src.constants import DEVICE
from src.ml.utils import collate_fn
from torch.utils.data import DataLoader
from src.exception import CustomException
from src.utils.main_utils import load_object
from src.ml.engine import train_one_epoch, evaluate
from src.entity.config_entity import ModelTrainerConfig
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from src.entity.artifact_entity import DataTransformationArtifacts, ModelTrainerArtifacts


class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig,
                 data_transformation_artifacts: DataTransformationArtifacts):
        """
        :param model_trainer_config: Configuration for model trainer
        :param data_transformation_artifacts: Artifacts for data transformation
        """
        self.model_trainer_config = model_trainer_config
        self.data_transformation_artifacts = data_transformation_artifacts

    def get_instance_segmentation_model(self):
        logging.info("Entered the get_instance_segmentation_model method of Model trainer class")
        try:
            logging.info("load an instance segmentation model pre-trained on COCO")
            model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

            logging.info("Get the number of input features for the classifier")
            in_features = model.roi_heads.box_predictor.cls_score.in_features

            logging.info("Replace the pre-trained head with a new one")
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.model_trainer_config.NUM_CLASSES)

            logging.info("Now get the number of input features for the mask classifier")
            in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels

            logging.info("And replace the mask predictor with a new one")
            model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                               self.model_trainer_config.HIDDEN_LAYER,
                                                               self.model_trainer_config.NUM_CLASSES)

            logging.info("Exited the get_instance_segmentation_model method of Model trainer class")
            return model
        except Exception as e:
            raise CustomException(e, sys) from e

    def construct_optimizer(self, model):
        logging.info("Entered the construct_optimizer method of Model trainer class")
        try:
            logging.info("Construct an optimizer")
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.SGD(params,
                                        lr=self.model_trainer_config.LR,
                                        momentum=self.model_trainer_config.MOMENTUM,
                                        weight_decay=self.model_trainer_config.WEIGHT_DECAY)
            logging.info("Exited the construct_optimizer method of Model trainer class")
            return optimizer
        except Exception as e:
            raise CustomException(e, sys) from e

    def get_lr(self, model):
        """
        A learning rate scheduler which decreases the learning rate by 10x every 3 epochs
        """
        logging.info("Entered the get_lr method of Model trainer class")
        try:
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.construct_optimizer(model),
                                                           step_size=self.model_trainer_config.STEP_SIZE,
                                                           gamma=self.model_trainer_config.GAMMA)
            logging.info("Exited the get_lr method of Model trainer class")
            return lr_scheduler
        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_model_trainer(self) -> ModelTrainerArtifacts:
        """
        Method Name :   initiate_model_trainer
        Description :   This function initiates a model trainer steps

        Output      :   Returns model trainer artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logging.info("Entered the initiate_model_trainer method of Model trainer class")

            train_dataset = load_object(self.data_transformation_artifacts.transformed_train_object)
            test_dataset = load_object(self.data_transformation_artifacts.transformed_test_object)
            logging.info("Loaded train and test dataset from data transformation artifacts")

            train_loader = DataLoader(train_dataset,
                                      batch_size=self.model_trainer_config.TRAIN_BATCH,
                                      shuffle=True,
                                      num_workers=self.model_trainer_config.NUM_WORKERS,
                                      collate_fn=collate_fn)

            test_loader = DataLoader(test_dataset,
                                     batch_size=self.model_trainer_config.TEST_BATCH,
                                     shuffle=False,
                                     num_workers=self.model_trainer_config.NUM_WORKERS,
                                     collate_fn=collate_fn)

            logging.info("Defined training and testing data loaders")

            model = self.get_instance_segmentation_model()
            logging.info("Got the instance segmentation model")
            model.to(DEVICE)

            optimizer = self.construct_optimizer(model)
            lr_scheduler = self.get_lr(model)

            logging.info(f"let's train it for {self.model_trainer_config.EPOCHS} epochs")
            for epoch in range(self.model_trainer_config.EPOCHS):
                logging.info("Train for one epoch, printing every 10 iterations")
                train_one_epoch(model, optimizer, train_loader, DEVICE, epoch, print_freq=10)
                logging.info("Update the learning rate")
                lr_scheduler.step()
                logging.info("Evaluate on the test dataset")
                evaluate(model, test_loader, device=DEVICE)
            logging.info(f"Training completed for {self.model_trainer_config.EPOCHS} epochs")

            os.makedirs(self.model_trainer_config.MODEL_TRAINER_ARTIFACTS_DIR, exist_ok=True)
            torch.save(model, self.model_trainer_config.TRAINED_MODEL_PATH)
            logging.info(f"saved trained model at {self.model_trainer_config.TRAINED_MODEL_PATH}")

            model_trainer_artifacts = ModelTrainerArtifacts(
                trained_model_path=self.model_trainer_config.TRAINED_MODEL_PATH)
            logging.info(f"Model trainer artifact: {model_trainer_artifacts}")

            logging.info("Exited the initiate_model_trainer method of Model trainer class")
            return model_trainer_artifacts

        except Exception as e:
            raise CustomException(e, sys) from e
