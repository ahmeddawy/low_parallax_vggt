# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import os.path as osp
import random
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from data.base_dataset import BaseDataset
from data.dataset_util import read_image_cv2, threshold_depth_map


def _quat_to_rotmat(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """Convert COLMAP quaternion (qw, qx, qy, qz) to a 3x3 rotation matrix."""
    norm = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm <= 0:
        return np.eye(3, dtype=np.float32)
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm

    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float32,
    )


class ColmapDataset(BaseDataset):
    """
    Dataset for folders organized as:
      <DATASET_DIR>/<sequence_id>/
        images/frame_XXXXXX.jpg
        colmap/cameras.txt
        colmap/images.txt
        <depth_dir_name>/depths.npy OR <depth_dir_name>/*_depths.npz
    """

    FRAME_RE = re.compile(r"frame_(\d+)")

    def __init__(
        self,
        common_conf=None,
        common_config=None,
        split: str = "train",
        DATASET_DIR: str = None,
        sequence_list_file: str = None,
        split_list_dir: str = None,
        min_num_images: int = 24,
        len_train: int = 100000,
        len_test: int = 10000,
        depth_key: str = "depths",
        depth_dir_name: str = "depth",
        depth_min_percentile: float = -1,
        depth_max_percentile: float = 99,
        depth_max: float = -1,
        auto_convert_npz_to_npy: bool = False,
    ):
        if common_conf is None:
            common_conf = common_config
        if common_conf is None:
            raise ValueError("common_conf/common_config must be provided.")

        super().__init__(common_conf=common_conf)

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img

        if split == "train":
            self.len_train = len_train
        elif split in ["test", "val"]:
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}")

        if DATASET_DIR is None:
            raise ValueError("DATASET_DIR must be specified.")
        if not osp.isdir(DATASET_DIR):
            raise ValueError(f"DATASET_DIR does not exist: {DATASET_DIR}")
        self.dataset_dir = DATASET_DIR

        self.min_num_images = min_num_images # Filter out sequences with fewer than min_num_images frames
        self.depth_key = depth_key # The key inside a .npz file to load the depth array. 
        self.depth_dir_name = depth_dir_name # The subfolder name where depths.npy lives.
        self.depth_min_percentile = depth_min_percentile # Clips depth values below this percentile. -1 means disabled 
        self.depth_max_percentile = depth_max_percentile # Clips the top 1% of depth values as outliers. 
        self.depth_max = depth_max # Hard cap on depth values in meters. -1 means disabled. 
        self.auto_convert_npz_to_npy = auto_convert_npz_to_npy

        self.sequence_frames: Dict[str, List[dict]] = {}     # seq_name → [{image_path, depth_idx, extrinsics, intrinsics}, ...]
        self.sequence_depth_paths: Dict[str, str] = {}       # seq_name → "/path/to/masked_depth/depths.npy"
        self.depth_arrays_cache: Dict[str, np.ndarray] = {}  # depth file path → depth array

        requested_sequences = self._load_sequence_names(
            split=split,
            sequence_list_file=sequence_list_file,
            split_list_dir=split_list_dir,
        )

        for seq_name in requested_sequences:
            seq_dir = osp.join(self.dataset_dir, seq_name)
            sequence_info = self._build_sequence_info(seq_name, seq_dir)
            if sequence_info is None:
                continue

            frames, depth_path = sequence_info
            if len(frames) < self.min_num_images:
                continue

            self.sequence_frames[seq_name] = frames
            self.sequence_depth_paths[seq_name] = depth_path

        self.sequence_list = sorted(list(self.sequence_frames.keys()))
        self.sequence_list_len = len(self.sequence_list)

        if self.sequence_list_len == 0:
            raise RuntimeError(
                f"No valid sequences found in {self.dataset_dir}. "
                f"Check colmap/{self.depth_dir_name} files and split lists."
            )

        if self.debug:
            self.sequence_list = self.sequence_list[:1]
            self.sequence_list_len = len(self.sequence_list)

        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: COLMAP sequences: {self.sequence_list_len}")
        logging.info(f"{status}: COLMAP dataset length: {len(self)}")

    def _load_sequence_names(
        self,
        split: str,
        sequence_list_file: Optional[str],
        split_list_dir: Optional[str],
    ) -> List[str]:
        if sequence_list_file is not None:
            if not osp.isfile(sequence_list_file):
                raise ValueError(f"sequence_list_file does not exist: {sequence_list_file}")
            with open(sequence_list_file, "r") as f:
                return [line.strip() for line in f if line.strip()]

        if split_list_dir is not None:
            split_file = osp.join(split_list_dir, f"{split}.txt")
            if not osp.isfile(split_file):
                raise ValueError(f"Split file not found: {split_file}")
            with open(split_file, "r") as f:
                return [line.strip() for line in f if line.strip()]

        candidates = []
        for name in sorted(os.listdir(self.dataset_dir)):
            seq_dir = osp.join(self.dataset_dir, name)
            if not osp.isdir(seq_dir):
                continue
            if (
                osp.isfile(osp.join(seq_dir, "colmap", "images.txt"))
                and osp.isfile(osp.join(seq_dir, "colmap", "cameras.txt"))
                and osp.isdir(osp.join(seq_dir, "images"))
                and osp.isdir(osp.join(seq_dir, self.depth_dir_name))
            ):
                candidates.append(name)
        return candidates

    def _build_sequence_info(
        self, seq_name: str, seq_dir: str
    ) -> Optional[Tuple[List[dict], str]]:
        colmap_images_path = osp.join(seq_dir, "colmap", "images.txt")
        colmap_cameras_path = osp.join(seq_dir, "colmap", "cameras.txt")
        depth_path = self._resolve_depth_path(seq_name, seq_dir)

        if depth_path is None:
            logging.warning(
                f"[{seq_name}] No supported depth file found under {self.depth_dir_name}/. Skipping."
            )
            return None

        camera_map = self._parse_colmap_cameras(colmap_cameras_path)
        images_meta = self._parse_colmap_images(colmap_images_path)
        if len(images_meta) == 0:
            logging.warning(f"[{seq_name}] No images parsed from COLMAP images.txt. Skipping.")
            return None

        depth_len = self._get_depth_length(depth_path)
        frames = []
        fallback_depth_idx = 0

        for image_id, camera_id, image_name, extrinsic in images_meta:
            image_path = osp.join(seq_dir, image_name)
            if not osp.isfile(image_path):
                image_path = osp.join(seq_dir, "images", osp.basename(image_name))
            if not osp.isfile(image_path):
                continue

            frame_match = self.FRAME_RE.search(osp.basename(image_name))
            if frame_match is not None:
                depth_idx = int(frame_match.group(1))
            else:
                depth_idx = fallback_depth_idx
                fallback_depth_idx += 1

            if depth_idx < 0 or depth_idx >= depth_len:
                continue
            if camera_id not in camera_map:
                continue

            frames.append(
                {
                    "image_id": image_id,
                    "image_path": image_path,
                    "depth_idx": depth_idx,
                    "extrinsics": extrinsic.astype(np.float32),
                    "intrinsics": camera_map[camera_id].astype(np.float32),
                }
            )

        frames.sort(key=lambda x: x["image_id"])
        if len(frames) == 0:
            logging.warning(f"[{seq_name}] No valid frame/depth matches found. Skipping.")
            return None
        return frames, depth_path

    def _resolve_depth_path(self, seq_name: str, seq_dir: str) -> Optional[str]:
        depth_dir = osp.join(seq_dir, self.depth_dir_name)
        if not osp.isdir(depth_dir):
            return None
        npy_candidates = [
            osp.join(depth_dir, "depths.npy"),
            osp.join(depth_dir, f"{seq_name}_depths.npy"),
        ]
        npz_candidates = [
            osp.join(depth_dir, "depths.npz"),
            osp.join(depth_dir, f"{seq_name}_depths.npz"),
        ]

        for path in npy_candidates:
            if osp.isfile(path):
                return path

        wildcard_npy = sorted(
            [
                osp.join(depth_dir, f)
                for f in os.listdir(depth_dir)
                if f.endswith("_depths.npy")
            ]
        )
        if wildcard_npy:
            return wildcard_npy[0]

        for path in npz_candidates:
            if osp.isfile(path):
                if self.auto_convert_npz_to_npy:
                    return self._convert_npz_to_npy(path)
                return path

        wildcard_npz = sorted(
            [
                osp.join(depth_dir, f)
                for f in os.listdir(depth_dir)
                if f.endswith("_depths.npz")
            ]
        )
        if wildcard_npz:
            if self.auto_convert_npz_to_npy:
                return self._convert_npz_to_npy(wildcard_npz[0])
            return wildcard_npz[0]

        return None

    def _convert_npz_to_npy(self, npz_path: str) -> str:
        npy_path = osp.splitext(npz_path)[0] + ".npy"
        if osp.isfile(npy_path):
            return npy_path

        with np.load(npz_path) as data:
            if self.depth_key not in data:
                raise KeyError(
                    f"Depth key '{self.depth_key}' not found in {npz_path}. "
                    f"Available keys: {list(data.files)}"
                )
            depth = data[self.depth_key]
        np.save(npy_path, depth.astype(np.float32))
        logging.info(f"Converted {npz_path} -> {npy_path}")
        return npy_path

    def _parse_colmap_cameras(self, cameras_txt_path: str) -> Dict[int, np.ndarray]:
        camera_map: Dict[int, np.ndarray] = {}
        with open(cameras_txt_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            camera_id = int(parts[0])
            model = parts[1]
            params = list(map(float, parts[4:]))

            if model in ["SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"]:
                fx = fy = params[0]
                cx = params[1]
                cy = params[2]
            elif model in ["PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"]:
                fx = params[0]
                fy = params[1]
                cx = params[2]
                cy = params[3]
            else:
                raise ValueError(
                    f"Unsupported camera model '{model}' in {cameras_txt_path}. "
                    "Add parsing for this model."
                )

            intrinsic = np.array(
                [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32
            )
            camera_map[camera_id] = intrinsic

        return camera_map

    def _parse_colmap_images(
        self, images_txt_path: str
    ) -> List[Tuple[int, int, str, np.ndarray]]:
        items: List[Tuple[int, int, str, np.ndarray]] = []
        with open(images_txt_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 10:
                continue

            image_id = int(parts[0])
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            camera_id = int(parts[8])
            image_name = parts[9]

            rot = _quat_to_rotmat(qw, qx, qy, qz)
            trans = np.array([tx, ty, tz], dtype=np.float32).reshape(3, 1)
            extrinsic = np.concatenate([rot, trans], axis=1)
            items.append((image_id, camera_id, image_name, extrinsic))

        items.sort(key=lambda x: x[0])
        return items

    def _get_depth_length(self, depth_path: str) -> int:
        if depth_path.endswith(".npy"):
            depth_arr = np.load(depth_path, mmap_mode="r")
            return int(depth_arr.shape[0])

        if depth_path.endswith(".npz"):
            with np.load(depth_path) as data:
                if self.depth_key not in data:
                    raise KeyError(
                        f"Depth key '{self.depth_key}' not found in {depth_path}. "
                        f"Available keys: {list(data.files)}"
                    )
                return int(data[self.depth_key].shape[0])

        raise ValueError(f"Unsupported depth file: {depth_path}")

    def _get_depth_array(self, seq_name: str) -> np.ndarray:
        if seq_name in self.depth_arrays_cache:
            return self.depth_arrays_cache[seq_name]

        depth_path = self.sequence_depth_paths[seq_name]
        if depth_path.endswith(".npy"):
            arr = np.load(depth_path, mmap_mode="r")
        elif depth_path.endswith(".npz"):
            with np.load(depth_path) as data:
                arr = data[self.depth_key].astype(np.float32)
        else:
            raise ValueError(f"Unsupported depth file: {depth_path}")

        self.depth_arrays_cache[seq_name] = arr
        return arr

    def get_data(
        self,
        seq_index: int = None,
        img_per_seq: int = None,
        seq_name: str = None,
        ids: list = None,
        aspect_ratio: float = 1.0,
    ) -> dict:
        if self.inside_random and self.training:
            seq_index = random.randint(0, self.sequence_list_len - 1)

        if seq_name is None:
            if seq_index is None:
                seq_index = 0
            seq_index = int(seq_index) % self.sequence_list_len
            seq_name = self.sequence_list[seq_index]

        frames = self.sequence_frames[seq_name]
        num_frames = len(frames)

        if ids is None:
            ids = np.random.choice(num_frames, img_per_seq, replace=self.allow_duplicate_img)

        if self.get_nearby:
            ids = self.get_nearby_ids(ids, num_frames)

        target_image_shape = self.get_target_shape(aspect_ratio)
        depth_array = self._get_depth_array(seq_name)

        images = []
        depths = []
        cam_points = []
        world_points = []
        point_masks = []
        extrinsics = []
        intrinsics = []
        original_sizes = []

        for i in ids:
            frame = frames[int(i)]
            image_path = frame["image_path"]
            image = read_image_cv2(image_path)
            depth_map = np.array(depth_array[frame["depth_idx"]], dtype=np.float32, copy=True)
            depth_map = threshold_depth_map(
                depth_map,
                max_percentile=self.depth_max_percentile,
                min_percentile=self.depth_min_percentile,
                max_depth=self.depth_max,
            )

            original_size = np.array(image.shape[:2])
            extri_opencv = np.array(frame["extrinsics"], dtype=np.float32)
            intri_opencv = np.array(frame["intrinsics"], dtype=np.float32)

            if image.shape[:2] != depth_map.shape[:2]:
                logging.warning(
                    f"Image/depth size mismatch in {seq_name} {osp.basename(image_path)}: "
                    f"{image.shape[:2]} vs {depth_map.shape[:2]}. Skipping frame."
                )
                continue

            (
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                world_coords_points,
                cam_coords_points,
                point_mask,
                _,
            ) = self.process_one_image(
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                original_size,
                target_image_shape,
                filepath=image_path,
            )

            images.append(image)
            depths.append(depth_map)
            extrinsics.append(extri_opencv)
            intrinsics.append(intri_opencv)
            cam_points.append(cam_coords_points)
            world_points.append(world_coords_points)
            point_masks.append(point_mask)
            original_sizes.append(original_size)

        if len(images) == 0:
            raise RuntimeError(f"No valid frames sampled for sequence {seq_name}")

        # Load GT tracks if available (e.g. AE plane tracks)
        tracks = None
        track_masks = None
        seq_dir = osp.join(self.dataset_dir, seq_name)
        tracks_path = osp.join(seq_dir, "tracks.npy")
        if osp.isfile(tracks_path):
            full_tracks = np.load(tracks_path)                      # (N_frames, N_tracks, 2)
            raw_tracks = full_tracks[ids]                           # (S, N_tracks, 2) in original pixel space

            # Transform track coordinates from original pixel space → model input space.
            # process_one_image applies resize + crop + optional scale aug, all of which
            # are captured in the change from original intrinsics to new intrinsics.
            # For a point (u, v) in original space:
            #   u_new = (u - cx_orig) / fx_orig * fx_new + cx_new
            #   v_new = (v - cy_orig) / fy_orig * fy_new + cy_new
            tracks = np.empty_like(raw_tracks)
            for frame_slot, frame_idx in enumerate(ids):
                orig_K = frames[int(frame_idx)]["intrinsics"]       # 3×3 original COLMAP intrinsics
                new_K  = intrinsics[frame_slot]                     # 3×3 after process_one_image
                fx_o, fy_o = orig_K[0, 0], orig_K[1, 1]
                cx_o, cy_o = orig_K[0, 2], orig_K[1, 2]
                fx_n, fy_n = new_K[0, 0],  new_K[1, 1]
                cx_n, cy_n = new_K[0, 2],  new_K[1, 2]
                u = raw_tracks[frame_slot, :, 0]
                v = raw_tracks[frame_slot, :, 1]
                tracks[frame_slot, :, 0] = (u - cx_o) / fx_o * fx_n + cx_n
                tracks[frame_slot, :, 1] = (v - cy_o) / fy_o * fy_n + cy_n

            track_masks_path = osp.join(seq_dir, "track_masks.npy")
            if osp.isfile(track_masks_path):
                full_masks = np.load(track_masks_path)              # (N_frames, N_tracks)
                track_masks = full_masks[ids]                       # (S, N_tracks)
            else:
                track_masks = np.ones(tracks.shape[:2], dtype=bool) # all visible

        set_name = "colmap"
        batch = {
            "seq_name": f"{set_name}_{seq_name}",
            "ids": np.asarray(ids),
            "frame_num": len(extrinsics),
            "images": images,
            "depths": depths,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "cam_points": cam_points,
            "world_points": world_points,
            "point_masks": point_masks,
            "original_sizes": original_sizes,
            "tracks": tracks,
            "tracks_orig": raw_tracks if tracks is not None else None,  # (S, N, 2) original pixel space
            "track_masks": track_masks,
        }
        return batch
