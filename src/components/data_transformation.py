import os
import sys
import torch
from src.logger import logging
from src.ml import transforms as T
from src.exception import CustomException
from src.utils.main_utils import save_object
from src.components.data_preparation import PennFudanDataset
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataIngestionArtifacts, DataTransformationArtifacts


class DataTransformation:
    def __init__(self, data_transformation_config: DataTransformationConfig,
                 data_ingestion_artifacts: DataIngestionArtifacts):
        """
        :param data_transformation_config: Configuration for data transformation
        :param data_ingestion_artifacts: Artifacts for data ingestion
        """
        self.data_transformation_config = data_transformation_config
        self.data_ingestion_artifacts = data_ingestion_artifacts

    def get_transform(self, train):
        try:
            logging.info("Entered the get_transform method of Data transformation class")
            transforms = []
            logging.info("Converts the image, a PIL image, into a PyTorch Tensor")
            transforms.append(T.ToTensor())
            if train:
                logging.info("During training, randomly flip the training images"
                             "and ground-truth for data augmentation")
                transforms.append(T.RandomHorizontalFlip(self.data_transformation_config.PROB))
            logging.info("Exited the get_transform method of Data transformation class")
            return T.Compose(transforms)
        except Exception as e:
            raise CustomException(e, sys) from e

    def split_into_train_and_test(self, dataset, dataset_test):
        try:
            logging.info("Entered the split_into_train_and_test method of Data transformation class")
            torch.manual_seed(1)
            indices = torch.randperm(len(dataset)).tolist()
            dataset = torch.utils.data.Subset(dataset, indices[:-50])
            dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])
            logging.info("Existed the split_into_train_and_test method of Data transformation class")
            return dataset, dataset_test
        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_data_transformation(self) -> DataTransformationArtifacts:
        """
        Method Name :   initiate_data_transformation
        Description :   This function initiates a data transformation steps

        Output      :   Returns data transformation artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logging.info("Entered the initiate_data_transformation method of Data transformation class")

            dataset = PennFudanDataset(root=self.data_ingestion_artifacts.dataset_path,
                                       transforms=self.get_transform(train=True))
            logging.info(f"Training dataset prepared")

            dataset_test = PennFudanDataset(root=self.data_ingestion_artifacts.dataset_path,
                                            transforms=self.get_transform(train=False))
            logging.info(f"Testing dataset prepared")

            train_dataset, test_dataset = self.split_into_train_and_test(dataset, dataset_test)
            logging.info("Split dataset into train and test")

            save_object(self.data_transformation_config.TRAIN_TRANSFORM_OBJECT_FILE_PATH, train_dataset)
            save_object(self.data_transformation_config.TEST_TRANSFORM_OBJECT_FILE_PATH, test_dataset)
            logging.info("Saved the train and test transformed object")

            data_transformation_artifact = DataTransformationArtifacts(
                transformed_train_object=self.data_transformation_config.TRAIN_TRANSFORM_OBJECT_FILE_PATH,
                transformed_test_object=self.data_transformation_config.TEST_TRANSFORM_OBJECT_FILE_PATH)
            logging.info(f"Data transformation artifact: {data_transformation_artifact}")

            logging.info("Exited the initiate_data_transformation method of Data transformation class")
            return data_transformation_artifact

        except Exception as e:
            raise CustomException(e, sys) from e
