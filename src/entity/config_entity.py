import os
from src.constants import *
from dataclasses import dataclass
from src.utils.main_utils import read_yaml_file


@dataclass
class DataIngestionConfig:
    def __init__(self):
        self.config = read_yaml_file(CONFIG_PATH)
        self.BUCKET_NAME: str = self.config['data_ingestion_config']["bucket_name"]
        self.ZIP_FILE_NAME: str = self.config['data_ingestion_config']["zip_file_name"]
        self.DATA_INGESTION_ARTIFACTS_DIR: str = os.path.join(os.getcwd(), ARTIFACTS_DIR, DATA_INGESTION_ARTIFACTS_DIR)
        self.ZIP_FILE_PATH: str = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR, self.ZIP_FILE_NAME)


@dataclass
class DataTransformationConfig:
    def __init__(self):
        self.config = read_yaml_file(CONFIG_PATH)
        self.PROB: float = self.config["data_transformation_config"]["prob"]
        self.DATA_TRANSFORMATION_ARTIFACTS_DIR: str = os.path.join(os.getcwd(), ARTIFACTS_DIR,
                                                                   DATA_TRANSFORMATION_ARTIFACTS_DIR)
        self.TRAIN_TRANSFORM_OBJECT_FILE_PATH: str = os.path.join(self.DATA_TRANSFORMATION_ARTIFACTS_DIR,
                                                                  DATA_TRANSFORMATION_TRAIN_FILE_NAME)
        self.TEST_TRANSFORM_OBJECT_FILE_PATH: str = os.path.join(self.DATA_TRANSFORMATION_ARTIFACTS_DIR,
                                                                 DATA_TRANSFORMATION_TEST_FILE_NAME)


@dataclass
class ModelTrainerConfig:
    def __init__(self):
        self.config = read_yaml_file(CONFIG_PATH)
        self.LR: float = self.config['model_trainer_config']['lr']
        self.MOMENTUM: float = self.config['model_trainer_config']['momentum']
        self.WEIGHT_DECAY: float = self.config['model_trainer_config']['weight_decay']
        self.STEP_SIZE: int = self.config['model_trainer_config']['step_size']
        self.GAMMA: float = self.config['model_trainer_config']['gamma']
        self.EPOCHS: int = self.config['model_trainer_config']['epochs']
        self.TEST_BATCH: int = self.config['model_trainer_config']['test_batch']
        self.TRAIN_BATCH: int = self.config['model_trainer_config']['train_batch']
        self.NUM_CLASSES: int = self.config['model_trainer_config']['num_classes']
        self.NUM_WORKERS: int = self.config['model_trainer_config']['num_workers']
        self.HIDDEN_LAYER: int = self.config['model_trainer_config']['hidden_layer']
        self.MODEL_TRAINER_ARTIFACTS_DIR: str = os.path.join(os.getcwd(), ARTIFACTS_DIR,
                                                             MODEL_TRAINER_ARTIFACTS_DIR)
        self.TRAINED_MODEL_PATH: str = os.path.join(self.MODEL_TRAINER_ARTIFACTS_DIR,
                                                    TRAINED_MODEL_PATH)


@dataclass
class ModelEvaluationConfig:
    def __init__(self):
        self.config = read_yaml_file(CONFIG_PATH)
        self.MODEL_NAME: str = MODEL_NAME
        self.BUCKET_NAME: str = self.config['model_evaluation_config']["bucket_name"]
        self.BATCH_SIZE: int = self.config['model_evaluation_config']["batch_size"]
        self.NUM_WORKERS: int = self.config['model_evaluation_config']["num_workers"]
        self.MODEL_EVALUATION_ARTIFACTS_DIR: str = os.path.join(os.getcwd(), ARTIFACTS_DIR,
                                                                MODEL_EVALUATION_ARTIFACTS_DIR)
        self.BEST_MODEL_DIR: str = os.path.join(MODEL_EVALUATION_ARTIFACTS_DIR, BEST_MODEL_DIR)


@dataclass
class ModelPusherConfig:
    def __init__(self):
        self.config = read_yaml_file(CONFIG_PATH)
        self.MODEL_NAME: str = MODEL_NAME
        self.BUCKET_NAME: str = self.config['model_pusher_config']["bucket_name"]


