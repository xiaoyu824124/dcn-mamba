# Author: Zixiang Zhao
# Last modified: 2025-10-20

from src.dataset.base_two_modal_dataset import BaseTwoModalDataset, DatasetMode

infrared_visible_columns = {
    DatasetMode.TRAIN: ["ir", "rgb"],
    DatasetMode.EVAL: ["ir", "rgb"],
    DatasetMode.TEST: ["ir", "rgb"],
}


class InfraredVisibleDataset(BaseTwoModalDataset):
    def __init__(
        self,
        column_names=None,
        **kwargs,
    ):
        # `column_names` may be overridden from the dataset config, e.g. VTMOT
        # adds `rgb_gt` (the aligned visible frames) so that the mono-modal
        # warm-up and the cross-modal task can share one supervised target.
        if column_names is not None:
            column_names = {
                mode if isinstance(mode, DatasetMode) else DatasetMode(str(mode)):
                list(columns)
                for mode, columns in dict(column_names).items()
            }
        else:
            column_names = infrared_visible_columns
        super().__init__(
            filename_col_names=column_names,
            **kwargs,
        )
