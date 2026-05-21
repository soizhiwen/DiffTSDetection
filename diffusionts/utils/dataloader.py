import torch
from utils.io_utils import instantiate_from_config


def build_dataloader(config):
    batch_size = config["dataloader"]["batch_size"]
    jud = config["dataloader"]["shuffle"]
    dataset = instantiate_from_config(config["dataloader"]["train_dataset"])

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=jud,
        num_workers=0,
        pin_memory=True,
        sampler=None,
        drop_last=jud,
    )

    dataload_info = {"dataloader": dataloader, "dataset": dataset}
    return dataload_info


def build_dataloader_recon(config):
    batch_size = config["dataloader"]["sample_size"]
    dataset = instantiate_from_config(config["dataloader"]["test_dataset"])

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        sampler=None,
        drop_last=False,
    )

    dataload_info = {"dataloader": dataloader, "dataset": dataset}
    return dataload_info


def build_dataloader_impute(config, missing_ratio):
    batch_size = config["dataloader"]["sample_size"]
    config["dataloader"]["test_dataset"]["params"]["missing_ratio"] = missing_ratio
    dataset = instantiate_from_config(config["dataloader"]["test_dataset"])

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        sampler=None,
        drop_last=False,
    )

    dataload_info = {"dataloader": dataloader, "dataset": dataset}
    return dataload_info
