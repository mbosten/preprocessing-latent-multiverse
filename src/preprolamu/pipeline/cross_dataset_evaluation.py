from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


# def _eval_model_on_universe(
#     universe: Universe,
#     split="test",
# ):
#     ds_cfg = load_dataset_config(universe.dataset_id)
#     label_col = ds_cfg.label_column

#     df = pd.read_parquet(universe.io.paths.preprocessed(split=split))
