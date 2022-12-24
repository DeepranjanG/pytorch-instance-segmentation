from asyncio import tasks
import json
from textwrap import dedent
import pendulum
import os


# The DAG object; we'll need this to instantiate a DAG
from airflow import DAG
training_pipeline = None
# Operators; we need this to operate!
from airflow.operators.python import PythonOperator

# [END imporETL DAG tutorial_prediction',
# [START default_args]
# These args will get passed on to each operator
# You can override them on a per-task basis during operator initialization
with DAG(
        'segmentation',
        default_args={'retries': 2},
        # [END default_args]
        description='Custom Image Segmentation',
        schedule_interval="@weekly",
        start_date=pendulum.datetime(2022, 12, 16, tz="UTC"),
        catchup=False,
        tags=['example'],
) as dag:
    # [END instantiate_dag]
    # [START documentation]
    dag.doc_md = __doc__
    # [END documentation]

    # [START extract_function]
    from src.pipeline.training import TrainingPipeline
    training_pipeline = TrainingPipeline()

    def data_ingestion(**kwargs):
        ti = kwargs['ti']
        data_ingestion_artifacts = training_pipeline.start_data_ingestion()
        print(data_ingestion_artifacts)
        ti.xcom_push('data_ingestion_artifacts', data_ingestion_artifacts.to_dict())

    def data_transformation(**kwargs):
        from src.entity.artifact_entity import DataIngestionArtifacts
        ti = kwargs['ti']
        data_ingestion_artifacts = ti.xcom_pull(task_ids="data_ingestion", key="data_ingestion_artifacts")
        data_ingestion_artifacts = DataIngestionArtifacts(**(data_ingestion_artifacts))
        data_transformation_artifacts = training_pipeline.start_data_transformation(
            data_ingestion_artifacts=data_ingestion_artifacts)
        ti.xcom_push('data_transformation_artifacts', data_transformation_artifacts.to_dict())

    def model_trainer(**kwargs):
        from src.entity.artifact_entity import DataTransformationArtifacts
        ti = kwargs['ti']
        data_transformation_artifacts = ti.xcom_pull(task_ids="data_transformation", key="data_transformation_artifacts")
        data_transformation_artifacts = DataTransformationArtifacts(**(data_transformation_artifacts))
        model_trainer_artifacts = training_pipeline.start_model_trainer(
            data_transformation_artifacts=data_transformation_artifacts)
        ti.xcom_push('model_trainer_artifacts', model_trainer_artifacts.to_dict())

    def model_evaluation(**kwargs):
        from src.entity.artifact_entity import ModelTrainerArtifacts, DataTransformationArtifacts
        ti = kwargs['ti']
        data_transformation_artifacts = ti.xcom_pull(task_ids="data_transformation", key="data_transformation_artifacts")
        data_transformation_artifacts = DataTransformationArtifacts(**(data_transformation_artifacts))

        model_trainer_artifacts = ti.xcom_pull(task_ids="model_trainer", key="model_trainer_artifacts")
        model_trainer_artifacts = ModelTrainerArtifacts(**(model_trainer_artifacts))

        model_evaluation_artifacts = training_pipeline.start_model_evaluation(
            model_trainer_artifacts=model_trainer_artifacts,
            data_transformation_artifacts=data_transformation_artifacts)
        ti.xcom_push('model_evaluation_artifacts', model_evaluation_artifacts.to_dict())


    def model_pusher(**kwargs):
        from src.entity.artifact_entity import ModelTrainerArtifacts
        ti = kwargs['ti']
        model_trainer_artifacts = ti.xcom_pull(task_ids="model_trainer", key="model_trainer_artifacts")
        model_trainer_artifacts = ModelTrainerArtifacts(**(model_trainer_artifacts))

        model_pusher_artifacts = training_pipeline.start_model_pusher(
            model_trainer_artifacts=model_trainer_artifacts)
        ti.xcom_push('model_pusher_artifacts', model_pusher_artifacts.to_dict())


    data_ingestion = PythonOperator(
        task_id='data_ingestion',
        python_callable=data_ingestion,
    )

    data_transformation = PythonOperator(
        task_id="data_transformation",
        python_callable=data_transformation
    )

    model_trainer = PythonOperator(
        task_id="model_trainer",
        python_callable=model_trainer
    )

    model_evaluation = PythonOperator(
        task_id="model_evaluation",
        python_callable=model_evaluation
    )

    model_pusher = PythonOperator(
        task_id="model_pusher",
        python_callable=model_pusher
    )

    data_ingestion >> data_transformation >> model_trainer >> model_evaluation >> model_pusher
