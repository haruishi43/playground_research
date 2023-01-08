from fastai.vision.all import (
    aug_transforms,
    DataLoaders,
    get_image_files,
    ImageDataLoaders,
    imagenet_stats,
    Normalize,
    RandomResizedCrop,
    untar_data,
    URLs,
)


def get_caltech_101_dataloaders() -> DataLoaders:
    """
    Gets DataLoaders for the Caltech-101 dataset

    Returns (DataLoaders): DataLoaders for the Caltech-101 dataset
    """
    path = untar_data(
        url=URLs.CALTECH_101,
    )
    validation_percent = 0.2
    seed = 42
    item_transforms = RandomResizedCrop(
        size=256,
    )
    batch_transforms = [
        *aug_transforms(),
        Normalize.from_stats(*imagenet_stats),
    ]
    batch_size = 32

    dataloaders = ImageDataLoaders.from_folder(
        path=path,
        valid_pct=validation_percent,
        seed=seed,
        item_tfms=item_transforms,
        batch_tfms=batch_transforms,
        bs=batch_size,
    )
    return dataloaders


def get_cub_200_2011_dataloaders() -> DataLoaders:
    """
    Gets DataLoaders for the Caltech-UCSD Birds-200-2011 dataset

    Returns (DataLoaders): DataLoaders for the Caltech-UCSD Birds-200-2011
    dataset
    """
    path = untar_data(
        url=URLs.CUB_200_2011,
    )
    path = path / "CUB_200_2011/images"
    validation_percent = 0.2
    seed = 42
    item_transforms = RandomResizedCrop(
        size=256,
    )
    batch_transforms = [
        *aug_transforms(),
        Normalize.from_stats(*imagenet_stats),
    ]
    batch_size = 32

    dataloaders = ImageDataLoaders.from_folder(
        path=path,
        valid_pct=validation_percent,
        seed=seed,
        item_tfms=item_transforms,
        batch_tfms=batch_transforms,
        bs=batch_size,
    )
    return dataloaders


def get_oxford_iiit_pet_dataloaders() -> DataLoaders:
    """
    Gets DataLoaders for the Oxford-IIIT Pet dataset

    Returns (DataLoaders): DataLoaders for the Oxford-IIIT Pet Dataset dataset
    """
    path = untar_data(
        url=URLs.PETS,
    )
    path = path / "images"
    filenames = get_image_files(
        path=path,
    )
    regex_pattern = "^(.*)_\d+.jpg"
    validation_percent = 0.2
    seed = 42
    item_transforms = RandomResizedCrop(
        size=256,
    )
    batch_transforms = [
        *aug_transforms(),
        Normalize.from_stats(*imagenet_stats),
    ]
    batch_size = 32

    dls = ImageDataLoaders.from_name_re(
        path=path,
        fnames=filenames,
        pat=regex_pattern,
        valid_pct=validation_percent,
        seed=seed,
        item_tfms=item_transforms,
        batch_tfms=batch_transforms,
        bs=batch_size,
    )
    return dls


def get_food_101_dataloaders() -> DataLoaders:
    """
    Gets DataLoaders for the Food-101 dataset

    Returns (DataLoaders): DataLoaders for the Food-101 dataset
    """
    path = untar_data(URLs.FOOD)
    training_data_path = "train"
    validation_data_path = "valid"
    seed = 42
    item_transforms = RandomResizedCrop(size=256)
    batch_transforms = [
        *aug_transforms(),
        Normalize.from_stats(*imagenet_stats),
    ]
    batch_size = 32

    dataloaders = ImageDataLoaders.from_folder(
        path=path,
        train=training_data_path,
        valid=validation_data_path,
        seed=seed,
        item_tfms=item_transforms,
        batch_tfms=batch_transforms,
        bs=batch_size,
    )
    return dataloaders


def get_dataloaders(
    dataset_name: str,
) -> DataLoaders:
    """
    Gets DataLoaders from dataset name

    Args:
        dataset_name (str): Name of dataset (must be one of ['caltech_101',
        'cub_200_2011', 'oxford_iiit_pet', 'food_101'])

    Returns (DataLoaders): DataLoaders for the specified dataset
    """
    if dataset_name == "caltech_101":
        dataloaders = get_caltech_101_dataloaders()

    elif dataset_name == "cub_200_2011":
        dataloaders = get_cub_200_2011_dataloaders()

    elif dataset_name == "oxford_iiit_pet":
        dataloaders = get_oxford_iiit_pet_dataloaders()

    elif dataset_name == "food_101":
        dataloaders = get_food_101_dataloaders()

    return dataloaders
