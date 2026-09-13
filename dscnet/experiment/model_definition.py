"""Models-from-code entry point for the DSCNet MLflow pyfunc package."""

import mlflow

from dscnet.experiment.modeling import DscnetPythonModel


mlflow.models.set_model(DscnetPythonModel())
