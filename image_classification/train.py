from fastai.vision.all import (
    accuracy,
    DataLoaders,
    Learner,
    ranger,
)
from torch.nn import Module


def get_learner(
    dataloaders: DataLoaders,
    model: Module,
) -> Learner:
    """
    Gets Learner from DataLoaders and model

    Args:
        dataloaders (DataLoaders): DataLoaders for the Learner
        model (Module): Model for the Learner
    """
    opt_func = ranger
    metrics = accuracy

    learn = Learner(
        dls=dataloaders,
        model=model,
        opt_func=opt_func,
        metrics=metrics,
    ).to_fp16()
    return learn


def train(
    learn: Learner,
    n_epochs: int,
) -> float:
    """
    Trains Learner

    Args:
        learn (Learner): Learner to train
        n_epochs (int): Number of epochs

    Returns (float): Final validation accuracy
    """
    batch_size = learn.dls.train.bs
    learning_rate = (batch_size / 32) * 4e-4

    learn.fit_flat_cos(
        n_epoch=n_epochs,
        lr=learning_rate,
    )

    validation_loss, validation_accuracy = learn.validate()
    return validation_accuracy
