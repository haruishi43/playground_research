from fastai.vision.all import *
from pandas import read_csv
from torch.nn import Module
from time import time
from typing import Any, Callable, Tuple
from train import train


def get_callable_execution_time(
    callable: Callable,
    **callable_arguments,
) -> Tuple[float, Any]:
    """
    Gets execution time of a Callable

    Args:
            callable (Callable): Callable to time
            **callable_arguments: Arguments passed to the Callable

    Returns (Tuple[float, Any]): Execution time and the output of the Callable
    """
    start_time = time()
    callable_output = callable(**callable_arguments)
    end_time = time()

    execution_time = end_time - start_time
    return execution_time, callable_output


def get_n_parameters(
    model: Module,
) -> int:
    """
    Gets the number of parameters of a model

    Args:
            model (Module): Module to count the parameters of

    Returns (int): Number of parameters of the Module
    """
    n_parameters = 0
    for p in model.parameters():
        n_parameters += p.numel()
    return n_parameters


def log_values(
    **values_to_log,
) -> None:
    """
    Logs the passed arguments to log.csv

    Args:
            values_to_log: Values to log to log.csv
    """
    dataframe = read_csv("log.csv")

    row = [values_to_log[key] for key in values_to_log]
    dataframe.loc[len(dataframe)] = row

    dataframe.to_csv("log.csv", index=False)


def train_and_log(
    learn: Learner,
    n_epochs: int,
    dataset_name: str,
    model_name: str,
) -> None:
    """
    Trains a Learner and logs some arguments (e.g., time and accuracy)
    """
    execution_time, validation_accuracy = get_callable_execution_time(
        callable=train,
        learn=learn,
        n_epochs=n_epochs,
    )

    n_parameters = get_n_parameters(
        model=learn.model,
    )

    log_values(
        model_name=model_name,
        dataset_name=dataset_name,
        n_epochs=n_epochs,
        accuracy=validation_accuracy,
        execution_time=execution_time,
        n_parameters=n_parameters,
    )
