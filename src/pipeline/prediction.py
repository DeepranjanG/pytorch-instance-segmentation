import os
import io
import sys
import base64
from io import BytesIO
from PIL import Image
from src.constants import *
from src.logger import logging
from torchvision import transforms
from src.exception import CustomException
from src.utils.main_utils import read_yaml_file
from src.configuration.gcloud_syncer import GCloudSync


class PredictionPipeline:
    def __init__(self,):
        self.gcloud = GCloudSync()
        self.config = read_yaml_file(CONFIG_PATH)

    def image_loader(self, image_bytes):
        """load image, returns cuda tensor"""
        logging.info("Entered the image_loader method of PredictionPipeline class")
        try:
            # image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            image = Image.open(io.BytesIO(image_bytes))
            convert_tensor = transforms.ToTensor()
            tensor_image = convert_tensor(image)
            logging.info("Exited the image_loader method of PredictionPipeline class")
            return tensor_image

        except Exception as e:
            raise CustomException(e, sys) from e

    def get_model_from_gcloud(self) -> str:
        """
        Method Name :   predict
        Description :   This method predicts the image.

        Output      :   Predictions
        """
        logging.info("Entered the get_model_from_gcloud method of PredictionPipeline class")
        try:
            logging.info("Loading the best model from gcloud bucket")
            os.makedirs("artifacts/PredictModel", exist_ok=True)
            predict_model_path = os.path.join(os.getcwd(), "artifacts", "PredictModel")
            self.gcloud.sync_file_from_gcloud(self.config['prediction_pipeline_config']["bucket_name"],
                                              self.config['prediction_pipeline_config']["model_name"],
                                              predict_model_path)
            best_model_path = os.path.join(predict_model_path, self.config['prediction_pipeline_config']["model_name"])
            logging.info("Exited the get_model_from_gcloud method of PredictionPipeline class")
            return best_model_path

        except Exception as e:
            raise CustomException(e, sys) from e

    def prediction(self, best_model_path: str, image_tensor) -> float:
        logging.info("Entered the prediction method of PredictionPipeline class")
        try:
            model = torch.load(best_model_path, map_location=torch.device(DEVICE))
            model.eval()
            with torch.no_grad():
                prediction = model([image_tensor.to(DEVICE)])
                pred = prediction[0]['masks'][0, 0].mul(255).byte().cpu().numpy()

            transform = transforms.ToPILImage()
            img = transform(pred)
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue())

            logging.info("Exited the prediction method of PredictionPipeline class")
            return img_str

        except Exception as e:
            raise CustomException(e, sys) from e

    def run_pipeline(self, data):
        logging.info("Entered the run_pipeline method of PredictionPipeline class")
        try:
            image = self.image_loader(data)
            best_model_path: str = self.get_model_from_gcloud()
            detected_image = self.prediction(best_model_path, image)
            logging.info("Exited the run_pipeline method of PredictionPipeline class")
            return detected_image
        except Exception as e:
            raise CustomException(e, sys) from e
