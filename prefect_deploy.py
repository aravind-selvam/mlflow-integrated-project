from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule
from prefect.flow_runners import SubprocessFlowRunner
from datetime import timedelta
from prefect.orion.schemas.schedules import IntervalSchedule


DeploymentSpec(
    flow_location="train.py",
    name="used-car-prediction",
    schedule= IntervalSchedule(interval=timedelta(minutes=200)),
    flow_runner=SubprocessFlowRunner(),
    tags=["ml"]
)