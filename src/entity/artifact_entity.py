from dataclasses import dataclass


# Model ingestion artifacts
@dataclass
class DataIngestionArtifacts:
    dataset_path: str

    def to_dict(self):
        return self.__dict__


# Data transformation artifacts
@dataclass
class DataTransformationArtifacts:
    transformed_train_object: str
    transformed_test_object: str

    def to_dict(self):
        return self.__dict__


# Model trainer artifacts
@dataclass
class ModelTrainerArtifacts:
    trained_model_path: str

    def to_dict(self):
        return self.__dict__


# Model evaluation artifacts
@dataclass
class ModelEvaluationArtifacts:
    is_model_accepted: bool

    def to_dict(self):
        return self.__dict__


# Model pusher artifacts
@dataclass
class ModelPusherArtifacts:
    bucket_name: str

    def to_dict(self):
        return self.__dict__
