# Author: Zixiang Zhao
# Last modified: 2025-10-05
# --------------------------------------------------------------------------


import os
from omegaconf import OmegaConf

from .base_two_modal_dataset import BaseTwoModalDataset, DatasetMode
from .infrared_visible_dataset import InfraredVisibleDataset

dataset_name_class_dict = {
    "infrared_visible_dataset": InfraredVisibleDataset,
}


def get_ir_visible_dataset(
    cfg_data_split: OmegaConf, base_data_dir: str, mode: DatasetMode, **kwargs
) -> BaseTwoModalDataset:
    if cfg_data_split.class_name in dataset_name_class_dict:
        # Proper deep copy to avoid modifying original config
        cfg_dataset = OmegaConf.create(
            OmegaConf.to_container(cfg_data_split, resolve=True)
        )
        dataset_class = dataset_name_class_dict[cfg_dataset.pop("class_name")]
        dataset = dataset_class(
            mode=mode,
            dataset_dir=os.path.join(base_data_dir, cfg_dataset.pop("dir")),
            **cfg_dataset,
            **kwargs,
        )
    else:
        raise NotImplementedError

    return dataset
