from argparse import ArgumentParser

from data import get_dataloaders
from log import train_and_log
from model import get_model
from train import get_learner


def main(
    dataset_name: str,
    model_name: str,
    n_epochs: int,
) -> None:
    dataloaders = get_dataloaders(
        dataset_name=dataset_name,
    )

    n_classes = dataloaders.c

    model = get_model(
        model_name=model_name,
        n_classes=n_classes,
    )

    learn = get_learner(
        dataloaders=dataloaders,
        model=model,
    )

    train_and_log(
        learn=learn,
        n_epochs=n_epochs,
        dataset_name=dataset_name,
        model_name=model_name,
    )


parser = ArgumentParser()
parser.add_argument("--dataset_name", type=str)
parser.add_argument("--model_name", type=str)
parser.add_argument("--n_epochs", type=int, default=5)
args = parser.parse_args()
main(
    dataset_name=args.dataset_name,
    model_name=args.model_name,
    n_epochs=args.n_epochs,
)
