# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import logging
from typing import (
    Any,
    Dict,
    List,
)

import torch
import torch.nn as nn
import os
from iopath.common.file_io import g_pathmgr
from wcmatch import fnmatch





# ------------------------------------------------------------
# Glob‑matching flags (behave like the Unix shell) 
# ------------------------------------------------------------
GLOB_FLAGS = (
    fnmatch.CASE       # case‑sensitive
    | fnmatch.DOTMATCH # '*' also matches '.'
    | fnmatch.EXTMATCH # extended patterns like *(foo|bar)
    | fnmatch.SPLIT    # "pat1|pat2" works out‑of‑the‑box
)




class DDPCheckpointSaver:
    def __init__(
        self,
        checkpoint_folder: str,
        checkpoint_names: List[str],
        rank: int,
        epoch: int,
    ):
        super().__init__()
        self.checkpoint_folder = checkpoint_folder
        self.checkpoint_names = checkpoint_names
        self.worker_id = rank
        self.epoch = epoch

    def save_checkpoint(
        self,
        model: nn.Module,
        **kwargs: Any,
    ) -> None:
        checkpoint = dict(**kwargs)
        checkpoint["model"] = model.state_dict()

        if self.worker_id == 0:
            for ckpt_name in self.checkpoint_names:
                checkpoint_path = os.path.join(
                    self.checkpoint_folder, f"{ckpt_name}.pt"
                )
                logging.info(
                    f"Saving checkpoint at epoch {self.epoch} to {checkpoint_path}"
                )
                robust_torch_save(checkpoint, checkpoint_path)



def robust_torch_save(checkpoint: Dict[str, Any], checkpoint_path: str) -> None:
    """
    A more robust version of torch.save that works better with preemptions
    and corruptions if a job is preempted during save.

    Always saves to a local temp file first, then copies to the destination.
    This avoids corruption when writing large files to GCS FUSE mounts, where
    streaming writes can leave truncated files if the process is interrupted.
    """
    import tempfile
    import shutil

    # Save to a local temp file atomically first
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".pt")
    try:
        os.close(tmp_fd)
        torch.save(checkpoint, tmp_path)

        # Move the existing checkpoint to a backup location
        backup_checkpoint_path = checkpoint_path + ".bak"
        backup_checkpoint_path_saved = False
        if g_pathmgr.exists(checkpoint_path):
            if g_pathmgr.exists(backup_checkpoint_path):
                g_pathmgr.rm(backup_checkpoint_path)
            g_pathmgr.mv(checkpoint_path, backup_checkpoint_path)
            backup_checkpoint_path_saved = True

        # Copy the fully-written local file to the destination
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        shutil.copy2(tmp_path, checkpoint_path)

        # Remove the backup checkpoint
        if backup_checkpoint_path_saved:
            g_pathmgr.rm(backup_checkpoint_path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)