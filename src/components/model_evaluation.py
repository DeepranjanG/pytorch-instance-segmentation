import os
import sys
import torch
from src.logger import logging
from src.constants import DEVICE
from src.ml.engine import evaluate
from src.ml.utils import collate_fn
from torch.utils.data import DataLoader
from src.exception import CustomException
from src.utils.main_utils import load_object
from src.configuration.gcloud_syncer import GCloudSync
from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import ModelEvaluationArtifacts, ModelTrainerArtifacts, DataTransformationArtifacts


class ModelEvaluation:
    def __init__(self, model_evaluation_config: ModelEvaluationConfig,
                 model_trainer_artifacts: ModelTrainerArtifacts,
                 data_transformation_artifacts: DataTransformationArtifacts):
        """
        :param model_evaluation_config: Configuration for model evaluation
        :param data_transformation_artifacts: Output reference of data transformation artifact stage
        :param model_trainer_artifacts: Output reference of model trainer artifact stage
        """

        self.model_evaluation_config = model_evaluation_config
        self.model_trainer_artifacts = model_trainer_artifacts
        self.data_transformation_artifacts = data_transformation_artifacts
        self.gcloud = GCloudSync()

    def get_best_model_from_gcloud(self) -> str:
        """
        :return: Fetch best model from gcloud storage and store inside best model directory path
        """
        try:
            logging.info("Entered the get_best_model_from_gcloud method of Model Evaluation class")

            os.makedirs(self.model_evaluation_config.BEST_MODEL_DIR, exist_ok=True)

            self.gcloud.sync_file_from_gcloud(self.model_evaluation_config.BUCKET_NAME,
                                              self.model_evaluation_config.MODEL_NAME,
                                              self.model_evaluation_config.BEST_MODEL_DIR)
            best_model_path = os.path.join(self.model_evaluation_config.BEST_MODEL_DIR,
                                           self.model_evaluation_config.MODEL_NAME)
            logging.info("Exited the get_best_model_from_gcloud method of Model Evaluation class")
            return best_model_path
        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_model_evaluation(self) -> ModelEvaluationArtifacts:
        """
            Method Name :   initiate_model_evaluation
            Description :   This function is used to initiate all steps of the model evaluation

            Output      :   Returns model evaluation artifact
            On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered the Initiate Model Evaluation")
        try:
            logging.info("Loading validation data for model evaluation")
            test_dataset = load_object(self.data_transformation_artifacts.transformed_test_object)
            test_loader = DataLoader(test_dataset,
                                     batch_size=self.model_evaluation_config.BATCH_SIZE,
                                     num_workers=self.model_evaluation_config.NUM_WORKERS,
                                     shuffle=False,
                                     collate_fn=collate_fn)

            logging.info("Loading currently trained model")
            trained_model = torch.load(self.model_trainer_artifacts.trained_model_path, map_location=DEVICE)
            trained_model_loss = evaluate(trained_model, test_loader, DEVICE)

            logging.info("Fetch best model from gcloud storage")
            best_model_path = self.get_best_model_from_gcloud()

            logging.info("Check is best model present in the gcloud storage or not ?")
            if os.path.isfile(best_model_path) is False:
                is_model_accepted = True
                logging.info("glcoud storage model is false and currently trained model accepted is true")

            else:
                logging.info("Load best model fetched from gcloud storage")
                best_model = torch.load(best_model_path, map_location=DEVICE)
                best_model_loss = evaluate(best_model, test_loader, DEVICE)

                logging.info("Comparing loss between best_model_loss and trained_model_loss ? ")
                if best_model_loss > trained_model_loss:
                    is_model_accepted = True
                    logging.info("Trained model not accepted")
                else:
                    is_model_accepted = False
                    logging.info("Trained model accepted")

            model_evaluation_artifacts = ModelEvaluationArtifacts(is_model_accepted=is_model_accepted)
            return model_evaluation_artifacts

        except Exception as e:
            raise CustomException(e, sys) from e
