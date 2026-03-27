# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F

from dataclasses import dataclass
from vggt.utils.pose_enc import extri_intri_to_pose_encoding, pose_encoding_to_extri_intri
from train_utils.general import check_and_fix_inf_nan
from math import ceil, floor


@dataclass(eq=False)
class MultitaskLoss(torch.nn.Module):
    """
    Multi-task loss module that combines different loss types for VGGT.
    
    Supports:
    - Camera loss
    - Depth loss 
    - Point loss
    - Tracking loss
    """
    def __init__(self, camera=None, depth=None, point=None, track=None, reproj=None, plane_rigidity=None, **kwargs):
        super().__init__()
        # Loss configuration dictionaries for each task
        self.camera = camera
        self.depth = depth
        self.point = point
        self.track = track
        self.reproj = reproj
        self.plane_rigidity = plane_rigidity

    def forward(self, predictions, batch, visualize=False) -> torch.Tensor:
        """
        Compute the total multi-task loss.
        
        Args:
            predictions: Dict containing model predictions for different tasks
            batch: Dict containing ground truth data and masks
            
        Returns:
            Dict containing individual losses and total objective
        """
        total_loss = 0
        loss_dict = {}
        
        # Camera pose loss - if pose encodings are predicted AND camera loss is configured
        if self.camera is not None and "pose_enc_list" in predictions:
            camera_loss_dict = compute_camera_loss(predictions, batch, **self.camera)
            camera_loss = camera_loss_dict["loss_camera"] * self.camera["weight"]
            total_loss = total_loss + camera_loss
            loss_dict.update(camera_loss_dict)

        # Depth estimation loss - if depth maps are predicted AND depth loss is configured
        if self.depth is not None and "depth" in predictions:
            depth_loss_dict = compute_depth_loss(predictions, batch, **self.depth)
            depth_loss = (
                depth_loss_dict["loss_conf_depth"]
                + depth_loss_dict["loss_reg_depth"]
                + depth_loss_dict["loss_grad_depth"]
                + depth_loss_dict.get("loss_mask_conf_depth", 0.0)
            )
            depth_loss = depth_loss * self.depth["weight"]
            total_loss = total_loss + depth_loss
            loss_dict.update(depth_loss_dict)

        # 3D point reconstruction loss - if world points are predicted AND point loss is configured
        if self.point is not None and "world_points" in predictions:
            point_loss_dict = compute_point_loss(predictions, batch, **self.point)
            point_loss = (
                point_loss_dict["loss_conf_point"]
                + point_loss_dict["loss_reg_point"]
                + point_loss_dict["loss_grad_point"]
                + point_loss_dict.get("loss_mask_conf_point", 0.0)
            )
            point_loss = point_loss * self.point["weight"]
            total_loss = total_loss + point_loss
            loss_dict.update(point_loss_dict)

        # Tracking loss
        has_tracks = batch.get("tracks") is not None
        if "track_list" in predictions and self.track is not None and has_tracks:
            track_loss, vis_loss, conf_loss = compute_track_loss(
                predictions["track_list"],
                predictions["vis"],
                predictions.get("conf", None),
                batch,
                **self.track,
            )
            if track_loss is not None:
                total_track_loss = (
                    track_loss + vis_loss + conf_loss
                ) * self.track.get("weight", 1.0)
                total_loss = total_loss + total_track_loss
                loss_dict["loss_track"] = track_loss
                loss_dict["loss_vis_track"] = vis_loss
                loss_dict["loss_conf_track"] = conf_loss

        # Reprojection loss — jointly constrains predicted depth + poses via GT plane tracks
        if (
            self.reproj is not None
            and "pose_enc_list" in predictions
            and "depth" in predictions
            and has_tracks
        ):
            reproj_loss = compute_reprojection_loss(predictions, batch, **self.reproj)
            if reproj_loss is not None:
                total_loss = total_loss + reproj_loss * self.reproj.get("weight", 1.0)
                loss_dict["loss_reproj"] = reproj_loss

        # Plane rigidity loss — self-supervised cross-frame 3D consistency for rigid plane tracks
        if (
            self.plane_rigidity is not None
            and "pose_enc_list" in predictions
            and "depth" in predictions
            and has_tracks
        ):
            plane_result = compute_plane_rigidity_loss(
                predictions, batch, visualize=visualize, **self.plane_rigidity
            )
            if visualize:
                plane_loss, vis_image = plane_result if plane_result is not None else (None, None)
            else:
                plane_loss, vis_image = plane_result, None
            if plane_loss is not None:
                total_loss = total_loss + plane_loss * self.plane_rigidity.get("weight", 1.0)
                loss_dict["loss_plane_rigidity"] = plane_loss
            if vis_image is not None:
                loss_dict["vis_plane_fitting"] = vis_image

        loss_dict["objective"] = total_loss

        return loss_dict


def compute_camera_loss(
    pred_dict,              # predictions dict, contains pose encodings
    batch_data,             # ground truth and mask batch dict
    loss_type="l1",         # "l1" or "l2" loss
    gamma=0.6,              # temporal decay weight for multi-stage training
    pose_encoding_type="absT_quaR_FoV",
    weight_trans=1.0,       # weight for translation loss
    weight_rot=1.0,         # weight for rotation loss
    weight_focal=0.5,       # weight for focal length loss
    **kwargs
):
    # List of predicted pose encodings per stage
    pred_pose_encodings = pred_dict['pose_enc_list']
    # Binary mask for valid points per frame (B, N, H, W)
    point_masks = batch_data['point_masks']
    # Only consider frames with enough valid points (>100)
    valid_frame_mask = point_masks[:, 0].sum(dim=[-1, -2]) > 100
    # Number of prediction stages
    n_stages = len(pred_pose_encodings)

    # Get ground truth camera extrinsics and intrinsics
    gt_extrinsics = batch_data['extrinsics']
    gt_intrinsics = batch_data['intrinsics']
    image_hw = batch_data['images'].shape[-2:]

    # Encode ground truth pose to match predicted encoding format
    gt_pose_encoding = extri_intri_to_pose_encoding(
        gt_extrinsics, gt_intrinsics, image_hw, pose_encoding_type=pose_encoding_type
    )

    # Initialize loss accumulators for translation, rotation, focal length
    total_loss_T = total_loss_R = total_loss_FL = 0

    # Compute loss for each prediction stage with temporal weighting
    for stage_idx in range(n_stages):
        # Later stages get higher weight (gamma^0 = 1.0 for final stage)
        stage_weight = gamma ** (n_stages - stage_idx - 1)
        pred_pose_stage = pred_pose_encodings[stage_idx]

        if valid_frame_mask.sum() == 0:
            # If no valid frames, set losses to zero to avoid gradient issues
            loss_T_stage = (pred_pose_stage * 0).mean()
            loss_R_stage = (pred_pose_stage * 0).mean()
            loss_FL_stage = (pred_pose_stage * 0).mean()
        else:
            # Only consider valid frames for loss computation
            loss_T_stage, loss_R_stage, loss_FL_stage = camera_loss_single(
                pred_pose_stage[valid_frame_mask].clone(),
                gt_pose_encoding[valid_frame_mask].clone(),
                loss_type=loss_type
            )
        # Accumulate weighted losses across stages
        total_loss_T += loss_T_stage * stage_weight
        total_loss_R += loss_R_stage * stage_weight
        total_loss_FL += loss_FL_stage * stage_weight

    # Average over all stages
    avg_loss_T = total_loss_T / n_stages
    avg_loss_R = total_loss_R / n_stages
    avg_loss_FL = total_loss_FL / n_stages

    # Compute total weighted camera loss
    total_camera_loss = (
        avg_loss_T * weight_trans +
        avg_loss_R * weight_rot +
        avg_loss_FL * weight_focal
    )

    # Return loss dictionary with individual components
    return {
        "loss_camera": total_camera_loss,
        "loss_T": avg_loss_T,
        "loss_R": avg_loss_R,
        "loss_FL": avg_loss_FL
    }

def camera_loss_single(pred_pose_enc, gt_pose_enc, loss_type="l1"):
    """
    Computes translation, rotation, and focal loss for a batch of pose encodings.
    
    Args:
        pred_pose_enc: (N, D) predicted pose encoding
        gt_pose_enc: (N, D) ground truth pose encoding
        loss_type: "l1" (abs error) or "l2" (euclidean error)
    Returns:
        loss_T: translation loss (mean)
        loss_R: rotation loss (mean)
        loss_FL: focal length/intrinsics loss (mean)
    
    NOTE: The paper uses smooth l1 loss, but we found l1 loss is more stable than smooth l1 and l2 loss.
        So here we use l1 loss.
    """
    if loss_type == "l1":
        # Translation: first 3 dims; Rotation: next 4 (quaternion); Focal/Intrinsics: last dims
        loss_T = (pred_pose_enc[..., :3] - gt_pose_enc[..., :3]).abs()
        loss_R = (pred_pose_enc[..., 3:7] - gt_pose_enc[..., 3:7]).abs()
        loss_FL = (pred_pose_enc[..., 7:] - gt_pose_enc[..., 7:]).abs()
    elif loss_type == "l2":
        # L2 norm for each component
        loss_T = (pred_pose_enc[..., :3] - gt_pose_enc[..., :3]).norm(dim=-1, keepdim=True)
        loss_R = (pred_pose_enc[..., 3:7] - gt_pose_enc[..., 3:7]).norm(dim=-1)
        loss_FL = (pred_pose_enc[..., 7:] - gt_pose_enc[..., 7:]).norm(dim=-1)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    # Check/fix numerical issues (nan/inf) for each loss component
    loss_T = check_and_fix_inf_nan(loss_T, "loss_T")
    loss_R = check_and_fix_inf_nan(loss_R, "loss_R")
    loss_FL = check_and_fix_inf_nan(loss_FL, "loss_FL")

    # Clamp outlier translation loss to prevent instability, then average
    loss_T = loss_T.clamp(max=100).mean()
    loss_R = loss_R.mean()
    loss_FL = loss_FL.mean()

    return loss_T, loss_R, loss_FL


def masked_conf_penalty(conf, valid_mask, invalid_conf_weight=0.0, invalid_conf_target=1.0, invalid_conf_loss_type="l2"):
    """
    Explicitly push confidence LOW at invalid (masked/zeroed) pixels.

    At pixels excluded from the depth loss (dynamic objects, plane), the
    confidence head receives no gradient from the regression loss. This term
    fills that gap by directly supervising conf toward its minimum value
    (1.0 for expp1 activation which has range [1, ∞)).

    Args:
        conf:                (B, S, H, W) predicted confidence map
        valid_mask:          (B, S, H, W) bool — True where depth loss fires
        invalid_conf_weight: scalar weight for this penalty (e.g. 0.05)
        invalid_conf_target: target conf at invalid pixels (1.0 = minimum for expp1)
        invalid_conf_loss_type: "l2" or "l1"
    """
    if invalid_conf_weight <= 0:
        return (conf * 0.0).mean()

    invalid_mask = ~valid_mask.bool()
    if invalid_mask.sum() == 0:
        return (conf * 0.0).mean()

    invalid_conf = conf[invalid_mask]
    target       = torch.full_like(invalid_conf, fill_value=invalid_conf_target)

    if invalid_conf_loss_type == "l1":
        penalty = (invalid_conf - target).abs().mean()
    else:
        penalty = ((invalid_conf - target) ** 2).mean()

    penalty = check_and_fix_inf_nan(penalty, "loss_mask_conf")
    return invalid_conf_weight * penalty


def compute_point_loss(predictions, batch, gamma=1.0, alpha=0.2, gradient_loss_fn=None, valid_range=-1, **kwargs):
    """
    Compute point loss.

    Args:
        predictions: Dict containing 'world_points' and 'world_points_conf'
        batch: Dict containing ground truth 'world_points' and 'point_masks'
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
        gradient_loss_fn: Type of gradient loss to apply
        valid_range: Quantile range for outlier filtering
        invalid_conf_weight: Weight for masked-region conf penalty (default 0)
        invalid_conf_target: Target conf at masked pixels (default 1.0)
        invalid_conf_loss_type: "l2" or "l1" (default "l2")
    """
    invalid_conf_weight    = float(kwargs.get("invalid_conf_weight", 0.0))
    invalid_conf_target    = float(kwargs.get("invalid_conf_target", 1.0))
    invalid_conf_loss_type = kwargs.get("invalid_conf_loss_type", "l2")

    pred_points      = predictions['world_points']
    pred_points_conf = predictions['world_points_conf']
    gt_points        = batch['world_points']
    gt_points_mask   = batch['point_masks']

    gt_points = check_and_fix_inf_nan(gt_points, "gt_points")

    loss_mask_conf = masked_conf_penalty(
        conf=pred_points_conf,
        valid_mask=gt_points_mask,
        invalid_conf_weight=invalid_conf_weight,
        invalid_conf_target=invalid_conf_target,
        invalid_conf_loss_type=invalid_conf_loss_type,
    )

    if gt_points_mask.sum() < 100:
        dummy_loss = (0.0 * pred_points).mean()
        return {
            "loss_conf_point":      dummy_loss,
            "loss_reg_point":       dummy_loss,
            "loss_grad_point":      dummy_loss,
            "loss_mask_conf_point": loss_mask_conf,
        }

    loss_conf, loss_grad, loss_reg = regression_loss(
        pred_points, gt_points, gt_points_mask, conf=pred_points_conf,
        gradient_loss_fn=gradient_loss_fn, gamma=gamma, alpha=alpha, valid_range=valid_range,
    )

    return {
        "loss_conf_point":      loss_conf,
        "loss_reg_point":       loss_reg,
        "loss_grad_point":      loss_grad,
        "loss_mask_conf_point": loss_mask_conf,
    }


def compute_depth_loss(predictions, batch, gamma=1.0, alpha=0.2, gradient_loss_fn=None, valid_range=-1, **kwargs):
    """
    Compute depth loss.

    Args:
        predictions: Dict containing 'depth' and 'depth_conf'
        batch: Dict containing ground truth 'depths' and 'point_masks'
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
        gradient_loss_fn: Type of gradient loss to apply
        valid_range: Quantile range for outlier filtering
        invalid_conf_weight: Weight for masked-region conf penalty (default 0)
        invalid_conf_target: Target conf at masked pixels (default 1.0)
        invalid_conf_loss_type: "l2" or "l1" (default "l2")
    """
    invalid_conf_weight    = float(kwargs.get("invalid_conf_weight", 0.0))
    invalid_conf_target    = float(kwargs.get("invalid_conf_target", 1.0))
    invalid_conf_loss_type = kwargs.get("invalid_conf_loss_type", "l2")

    pred_depth      = predictions['depth']
    pred_depth_conf = predictions['depth_conf']
    # Clamp predicted depth before loss — prevents diff.pow(2) overflow to Inf in bfloat16
    # when the depth head outputs extreme values early in training.
    pred_depth = pred_depth.clamp(min=1e-3, max=1e3)

    gt_depth      = batch['depths']
    gt_depth      = check_and_fix_inf_nan(gt_depth, "gt_depth")
    gt_depth      = gt_depth[..., None]                    # (B, S, H, W, 1)
    gt_depth_mask = batch['point_masks'].clone()           # False at plane + dynamic object pixels

    loss_mask_conf = masked_conf_penalty(
        conf=pred_depth_conf,
        valid_mask=gt_depth_mask,
        invalid_conf_weight=invalid_conf_weight,
        invalid_conf_target=invalid_conf_target,
        invalid_conf_loss_type=invalid_conf_loss_type,
    )

    if gt_depth_mask.sum() < 100:
        dummy_loss = (0.0 * pred_depth).mean()
        return {
            "loss_conf_depth":      dummy_loss,
            "loss_reg_depth":       dummy_loss,
            "loss_grad_depth":      dummy_loss,
            "loss_mask_conf_depth": loss_mask_conf,
        }

    # NOTE: conf is passed into regression_loss so it applies to the gradient loss too
    loss_conf, loss_grad, loss_reg = regression_loss(
        pred_depth, gt_depth, gt_depth_mask, conf=pred_depth_conf,
        gradient_loss_fn=gradient_loss_fn, gamma=gamma, alpha=alpha, valid_range=valid_range,
    )

    return {
        "loss_conf_depth":      loss_conf,
        "loss_reg_depth":       loss_reg,
        "loss_grad_depth":      loss_grad,
        "loss_mask_conf_depth": loss_mask_conf,
    }


def regression_loss(pred, gt, mask, conf=None, gradient_loss_fn=None, gamma=1.0, alpha=0.2, valid_range=-1):
    """
    Core regression loss function with confidence weighting and optional gradient loss.
    
    Computes:
    1. gamma * ||pred - gt||^2 * conf - alpha * log(conf)
    2. Optional gradient loss
    
    Args:
        pred: (B, S, H, W, C) predicted values
        gt: (B, S, H, W, C) ground truth values
        mask: (B, S, H, W) valid pixel mask
        conf: (B, S, H, W) confidence weights (optional)
        gradient_loss_fn: Type of gradient loss ("normal", "grad", etc.)
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
        valid_range: Quantile range for outlier filtering
    
    Returns:
        loss_conf: Confidence-weighted loss
        loss_grad: Gradient loss (0 if not specified)
        loss_reg: Regular L2 loss
    """
    bb, ss, hh, ww, nc = pred.shape

    # Compute L2 distance between predicted and ground truth points.
    # Use sqrt(sum_sq + eps) instead of torch.norm to avoid 0/0=NaN in backward:
    # when bfloat16 rounds pred==gt exactly, norm(0)=0 and grad=0/0=NaN.
    # eps=1e-8 keeps gradient finite: d/d(diff) = diff / sqrt(sum_sq + eps) → 0 when diff→0.
    diff = gt[mask] - pred[mask]
    loss_reg = (diff.pow(2).sum(-1) + 1e-8).sqrt()
    loss_reg = check_and_fix_inf_nan(loss_reg, "loss_reg")

    # Confidence-weighted loss: gamma * loss * conf - alpha * log(conf)
    # This encourages the model to be confident on easy examples and less confident on hard ones
    loss_conf = gamma * loss_reg * conf[mask] - alpha * torch.log(conf[mask])
    loss_conf = check_and_fix_inf_nan(loss_conf, "loss_conf")
        
    # Initialize gradient loss
    loss_grad = 0

    # Prepare confidence for gradient loss if needed
    if "conf" in gradient_loss_fn:
        to_feed_conf = conf.reshape(bb*ss, hh, ww)
    else:
        to_feed_conf = None

    # Compute gradient loss if specified for spatial smoothness
    if "normal" in gradient_loss_fn:
        # Surface normal-based gradient loss
        loss_grad = gradient_loss_multi_scale_wrapper(
            pred.reshape(bb*ss, hh, ww, nc),
            gt.reshape(bb*ss, hh, ww, nc),
            mask.reshape(bb*ss, hh, ww),
            gradient_loss_fn=normal_loss,
            scales=3,
            conf=to_feed_conf,
        )
    elif "grad" in gradient_loss_fn:
        # Standard gradient-based loss
        loss_grad = gradient_loss_multi_scale_wrapper(
            pred.reshape(bb*ss, hh, ww, nc),
            gt.reshape(bb*ss, hh, ww, nc),
            mask.reshape(bb*ss, hh, ww),
            gradient_loss_fn=gradient_loss,
            conf=to_feed_conf,
        )

    # Process confidence-weighted loss
    if loss_conf.numel() > 0:
        # Filter out outliers using quantile-based thresholding
        if valid_range>0:
            loss_conf = filter_by_quantile(loss_conf, valid_range)

        loss_conf = check_and_fix_inf_nan(loss_conf, f"loss_conf_depth")
        loss_conf = loss_conf.mean()
    else:
        loss_conf = (0.0 * pred).mean()

    # Process regular regression loss
    if loss_reg.numel() > 0:
        # Filter out outliers using quantile-based thresholding
        if valid_range>0:
            loss_reg = filter_by_quantile(loss_reg, valid_range)

        loss_reg = check_and_fix_inf_nan(loss_reg, f"loss_reg_depth")
        loss_reg = loss_reg.mean()
    else:
        loss_reg = (0.0 * pred).mean()

    return loss_conf, loss_grad, loss_reg


def gradient_loss_multi_scale_wrapper(prediction, target, mask, scales=4, gradient_loss_fn = None, conf=None):
    """
    Multi-scale gradient loss wrapper. Applies gradient loss at multiple scales by subsampling the input.
    This helps capture both fine and coarse spatial structures.
    
    Args:
        prediction: (B, H, W, C) predicted values
        target: (B, H, W, C) ground truth values  
        mask: (B, H, W) valid pixel mask
        scales: Number of scales to use
        gradient_loss_fn: Gradient loss function to apply
        conf: (B, H, W) confidence weights (optional)
    """
    total = 0
    for scale in range(scales):
        step = pow(2, scale)  # Subsample by 2^scale

        total += gradient_loss_fn(
            prediction[:, ::step, ::step],
            target[:, ::step, ::step],
            mask[:, ::step, ::step],
            conf=conf[:, ::step, ::step] if conf is not None else None
        )

    total = total / scales
    return total


def normal_loss(prediction, target, mask, cos_eps=1e-8, conf=None, gamma=1.0, alpha=0.2):
    """
    Surface normal-based loss for geometric consistency.
    
    Computes surface normals from 3D point maps using cross products of neighboring points,
    then measures the angle between predicted and ground truth normals.
    
    Args:
        prediction: (B, H, W, 3) predicted 3D coordinates/points
        target: (B, H, W, 3) ground-truth 3D coordinates/points
        mask: (B, H, W) valid pixel mask
        cos_eps: Epsilon for numerical stability in cosine computation
        conf: (B, H, W) confidence weights (optional)
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
    """
    # Convert point maps to surface normals using cross products
    pred_normals, pred_valids = point_map_to_normal(prediction, mask, eps=cos_eps)
    gt_normals,   gt_valids   = point_map_to_normal(target,     mask, eps=cos_eps)

    # Only consider regions where both predicted and GT normals are valid
    all_valid = pred_valids & gt_valids  # shape: (4, B, H, W)

    # Early return if not enough valid points
    divisor = torch.sum(all_valid)
    if divisor < 10:
        return 0

    # Extract valid normals
    pred_normals = pred_normals[all_valid].clone()
    gt_normals = gt_normals[all_valid].clone()

    # Compute cosine similarity between corresponding normals
    dot = torch.sum(pred_normals * gt_normals, dim=-1)

    # Clamp dot product to [-1, 1] for numerical stability
    dot = torch.clamp(dot, -1 + cos_eps, 1 - cos_eps)

    # Compute loss as 1 - cos(theta), instead of arccos(dot) for numerical stability
    loss = 1 - dot

    # Return mean loss if we have enough valid points
    if loss.numel() < 10:
        return 0
    else:
        loss = check_and_fix_inf_nan(loss, "normal_loss")

        if conf is not None:
            # Apply confidence weighting
            conf = conf[None, ...].expand(4, -1, -1, -1)
            conf = conf[all_valid].clone()

            loss = gamma * loss * conf - alpha * torch.log(conf)
            return loss.mean()
        else:
            return loss.mean()


def gradient_loss(prediction, target, mask, conf=None, gamma=1.0, alpha=0.2):
    """
    Gradient-based loss. Computes the L1 difference between adjacent pixels in x and y directions.
    
    Args:
        prediction: (B, H, W, C) predicted values
        target: (B, H, W, C) ground truth values
        mask: (B, H, W) valid pixel mask
        conf: (B, H, W) confidence weights (optional)
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
    """
    # Expand mask to match prediction channels
    mask = mask[..., None].expand(-1, -1, -1, prediction.shape[-1])
    M = torch.sum(mask, (1, 2, 3))

    # Compute difference between prediction and target
    diff = prediction - target
    diff = torch.mul(mask, diff)

    # Compute gradients in x direction (horizontal)
    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    # Compute gradients in y direction (vertical)
    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    # Clamp gradients to prevent outliers
    grad_x = grad_x.clamp(max=100)
    grad_y = grad_y.clamp(max=100)

    # Apply confidence weighting if provided
    if conf is not None:
        conf = conf[..., None].expand(-1, -1, -1, prediction.shape[-1])
        conf_x = conf[:, :, 1:]
        conf_y = conf[:, 1:, :]

        grad_x = gamma * grad_x * conf_x - alpha * torch.log(conf_x)
        grad_y = gamma * grad_y * conf_y - alpha * torch.log(conf_y)

    # Sum gradients and normalize by number of valid pixels
    grad_loss = torch.sum(grad_x, (1, 2, 3)) + torch.sum(grad_y, (1, 2, 3))
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        grad_loss = torch.sum(grad_loss) / divisor

    return grad_loss


def point_map_to_normal(point_map, mask, eps=1e-6):
    """
    Convert 3D point map to surface normal vectors using cross products.
    
    Computes normals by taking cross products of neighboring point differences.
    Uses 4 different cross-product directions for robustness.
    
    Args:
        point_map: (B, H, W, 3) 3D points laid out in a 2D grid
        mask: (B, H, W) valid pixels (bool)
        eps: Epsilon for numerical stability in normalization
    
    Returns:
        normals: (4, B, H, W, 3) normal vectors for each of the 4 cross-product directions
        valids: (4, B, H, W) corresponding valid masks
    """
    with torch.cuda.amp.autocast(enabled=False):
        # Pad inputs to avoid boundary issues
        padded_mask = F.pad(mask, (1, 1, 1, 1), mode='constant', value=0)
        pts = F.pad(point_map.permute(0, 3, 1, 2), (1,1,1,1), mode='constant', value=0).permute(0, 2, 3, 1)

        # Get neighboring points for each pixel
        center = pts[:, 1:-1, 1:-1, :]   # B,H,W,3
        up     = pts[:, :-2,  1:-1, :]
        left   = pts[:, 1:-1, :-2 , :]
        down   = pts[:, 2:,   1:-1, :]
        right  = pts[:, 1:-1, 2:,   :]

        # Compute direction vectors from center to neighbors
        up_dir    = up    - center
        left_dir  = left  - center
        down_dir  = down  - center
        right_dir = right - center

        # Compute four cross products for different normal directions
        n1 = torch.cross(up_dir,   left_dir,  dim=-1)  # up x left
        n2 = torch.cross(left_dir, down_dir,  dim=-1)  # left x down
        n3 = torch.cross(down_dir, right_dir, dim=-1)  # down x right
        n4 = torch.cross(right_dir,up_dir,    dim=-1)  # right x up

        # Validity masks - require both direction pixels to be valid
        v1 = padded_mask[:, :-2,  1:-1] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 1:-1, :-2]
        v2 = padded_mask[:, 1:-1, :-2 ] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 2:,   1:-1]
        v3 = padded_mask[:, 2:,   1:-1] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 1:-1, 2:]
        v4 = padded_mask[:, 1:-1, 2:  ] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, :-2,  1:-1]

        # Stack normals and validity masks
        normals = torch.stack([n1, n2, n3, n4], dim=0)  # shape [4, B, H, W, 3]
        valids  = torch.stack([v1, v2, v3, v4], dim=0)  # shape [4, B, H, W]

        # Normalize normal vectors
        normals = F.normalize(normals, p=2, dim=-1, eps=eps)

    return normals, valids


def filter_by_quantile(loss_tensor, valid_range, min_elements=1000, hard_max=100):
    """
    Filter loss tensor by keeping only values below a certain quantile threshold.
    
    This helps remove outliers that could destabilize training.
    
    Args:
        loss_tensor: Tensor containing loss values
        valid_range: Float between 0 and 1 indicating the quantile threshold
        min_elements: Minimum number of elements required to apply filtering
        hard_max: Maximum allowed value for any individual loss
    
    Returns:
        Filtered and clamped loss tensor
    """
    if loss_tensor.numel() <= min_elements:
        # Too few elements, just return as-is
        return loss_tensor

    # Randomly sample if tensor is too large to avoid memory issues
    if loss_tensor.numel() > 100000000:
        # Flatten and randomly select 1M elements
        indices = torch.randperm(loss_tensor.numel(), device=loss_tensor.device)[:1_000_000]
        loss_tensor = loss_tensor.view(-1)[indices]

    # First clamp individual values to prevent extreme outliers
    loss_tensor = loss_tensor.clamp(max=hard_max)

    # Compute quantile threshold
    quantile_thresh = torch_quantile(loss_tensor.detach(), valid_range)
    quantile_thresh = min(quantile_thresh, hard_max)

    # Apply quantile filtering if enough elements remain
    quantile_mask = loss_tensor < quantile_thresh
    if quantile_mask.sum() > min_elements:
        return loss_tensor[quantile_mask]
    return loss_tensor


def torch_quantile(
    input,
    q,
    dim = None,
    keepdim: bool = False,
    *,
    interpolation: str = "nearest",
    out: torch.Tensor = None,
) -> torch.Tensor:
    """Better torch.quantile for one SCALAR quantile.

    Using torch.kthvalue. Better than torch.quantile because:
        - No 2**24 input size limit (pytorch/issues/67592),
        - Much faster, at least on big input sizes.

    Arguments:
        input (torch.Tensor): See torch.quantile.
        q (float): See torch.quantile. Supports only scalar input
            currently.
        dim (int | None): See torch.quantile.
        keepdim (bool): See torch.quantile. Supports only False
            currently.
        interpolation: {"nearest", "lower", "higher"}
            See torch.quantile.
        out (torch.Tensor | None): See torch.quantile. Supports only
            None currently.
    """
    # https://github.com/pytorch/pytorch/issues/64947
    # Sanitization: q
    try:
        q = float(q)
        assert 0 <= q <= 1
    except Exception:
        raise ValueError(f"Only scalar input 0<=q<=1 is currently supported (got {q})!")

    # Handle dim=None case
    if dim_was_none := dim is None:
        dim = 0
        input = input.reshape((-1,) + (1,) * (input.ndim - 1))

    # Set interpolation method
    if interpolation == "nearest":
        inter = round
    elif interpolation == "lower":
        inter = floor
    elif interpolation == "higher":
        inter = ceil
    else:
        raise ValueError(
            "Supported interpolations currently are {'nearest', 'lower', 'higher'} "
            f"(got '{interpolation}')!"
        )

    # Validate out parameter
    if out is not None:
        raise ValueError(f"Only None value is currently supported for out (got {out})!")

    # Compute k-th value
    k = inter(q * (input.shape[dim] - 1)) + 1
    out = torch.kthvalue(input, k, dim, keepdim=True, out=out)[0]

    # Handle keepdim and dim=None cases
    if keepdim:
        return out
    if dim_was_none:
        return out.squeeze()
    else:
        return out.squeeze(dim)

    return out


def compute_track_loss(coord_preds, vis_scores, conf_scores, batch, gamma=0.8, **kwargs):
    """Compute tracking losses over all refinement iterations.

    Args:
        coord_preds: list of (B, S, N, 2) predicted coordinates per iteration
        vis_scores:  (B, S, N) raw visibility logits
        conf_scores: (B, S, N) raw confidence logits, or None
        batch:       dict with "tracks" (B, S, N, 2) and "track_vis_mask" (B, S, N bool)
        gamma:       per-iteration loss weighting (later iterations weighted higher)

    Returns:
        track_loss, vis_loss, conf_loss  — three scalar tensors
    """
    gt_tracks = batch["tracks"]          # B, S, N, 2
    gt_track_vis_mask = batch["track_vis_mask"]  # B, S, N  (bool)

    # Trim to the number of query points the model actually predicted
    n_query = coord_preds[-1].shape[2]
    gt_tracks = gt_tracks[:, :, :n_query]
    gt_tracks = check_and_fix_inf_nan(gt_tracks, "gt_tracks", hard_max=None)
    gt_track_vis_mask = gt_track_vis_mask[:, :, :n_query]

    # Valid mask: only supervise tracks that are visible in the first (query) frame
    valids = torch.ones_like(gt_track_vis_mask)
    first_frame_visible = gt_track_vis_mask[:, 0, :]  # B, N
    valids = valids * first_frame_visible.unsqueeze(1)

    valids = valids.bool()  # ensure bool for indexing

    if not valids.any():
        return None, None, None

    track_loss = sequence_loss(
        flow_preds=coord_preds,
        flow_gt=gt_tracks,
        vis=gt_track_vis_mask,
        valids=valids,
        gamma=gamma,
    )

    vis_loss = F.binary_cross_entropy_with_logits(
        vis_scores[valids], gt_track_vis_mask[valids].float()
    )
    vis_loss = check_and_fix_inf_nan(vis_loss, "vis_loss", hard_max=None)

    if conf_scores is not None:
        gt_conf_mask = (gt_tracks - coord_preds[-1]).norm(dim=-1) < 3
        conf_loss = F.binary_cross_entropy_with_logits(
            conf_scores[valids], gt_conf_mask[valids].float()
        )
        conf_loss = check_and_fix_inf_nan(conf_loss, "conf_loss", hard_max=None)
    else:
        conf_loss = torch.tensor(0.0, device=vis_loss.device)

    return track_loss, vis_loss, conf_loss


def compute_reprojection_loss(predictions, batch, weight=1.0, **kwargs):
    """
    Reprojection loss that jointly constrains predicted depth and predicted poses
    using GT plane (AE) tracks.

    For each GT track point visible in frame 0:
      1. Sample predicted depth at that pixel location.
      2. Unproject to 3D world coords via predicted extrinsics/intrinsics of frame 0.
      3. Reproject into every other frame via predicted extrinsics/intrinsics.
      4. L1 error against GT track positions.

    Args:
        predictions: dict with "pose_enc_list" (list of B,S,9) and "depth" (B,S,H,W,1)
        batch:       dict with "tracks" (B,S,N,2) and "track_vis_mask" (B,S,N bool)
        weight:      scalar weight (absorbed by caller, kept for **kwargs compatibility)

    Returns:
        Scalar loss tensor, or None if no valid tracks.
    """
    if "tracks" not in batch or batch["tracks"] is None:
        return None

    gt_tracks = batch["tracks"]       # B, S, N, 2
    gt_vis    = batch["track_vis_mask"]  # B, S, N (bool)

    B, S, N, _ = gt_tracks.shape

    first_vis = gt_vis[:, 0, :]       # B, N — tracks visible in frame 0
    if not first_vis.any():
        return None

    image_hw = batch["images"].shape[-2:]   # (H, W)
    H, W = image_hw

    # Decode last-stage pose encoding → extrinsics (B,S,3,4), intrinsics (B,S,3,3)
    pred_pose_enc = predictions["pose_enc_list"][-1]   # B, S, 9
    pred_extrinsics, pred_intrinsics = pose_encoding_to_extri_intri(pred_pose_enc, image_hw)

    # Predicted depth for frame 0: (B, H, W)
    pred_depth_0 = predictions["depth"][:, 0, :, :, 0]   # B, H, W

    # GT track coords in frame 0: (B, N, 2)  [u=col, v=row]
    track0 = gt_tracks[:, 0, :, :]   # B, N, 2

    # --- Sample depth at track positions via bilinear interpolation ---
    norm_u = (track0[..., 0] / (W - 1)) * 2.0 - 1.0   # B, N
    norm_v = (track0[..., 1] / (H - 1)) * 2.0 - 1.0   # B, N
    grid   = torch.stack([norm_u, norm_v], dim=-1).unsqueeze(1)   # B, 1, N, 2

    sampled_depth = F.grid_sample(
        pred_depth_0.unsqueeze(1),   # B, 1, H, W
        grid,
        mode="bilinear",
        align_corners=True,
    )  # B, 1, 1, N
    sampled_depth = sampled_depth.squeeze(1).squeeze(1)   # B, N
    # Clamp depth to valid range — prevents Inf intermediate values in the
    # unproject pipeline whose backward would give Inf*0=NaN (IEEE 754),
    # corrupting aggregator/depth-head weights even when forward is sanitised.
    sampled_depth = sampled_depth.clamp(min=1e-3, max=1e3)

    # --- Unproject frame-0 tracks to world 3D ---
    K0 = pred_intrinsics[:, 0, :, :]   # B, 3, 3
    # Analytical K^{-1} for upper-triangular intrinsic matrix.
    # Avoids linalg.inv whose backward = -K^{-T} @ g @ K^{-T}, which explodes
    # when fx/fy is small (near-singular K) and corrupts the aggregator grads.
    fx0 = K0[:, 0, 0].clamp(min=1e-2)
    fy0 = K0[:, 1, 1].clamp(min=1e-2)
    cx0 = K0[:, 0, 2]
    cy0 = K0[:, 1, 2]
    K0_inv = torch.zeros_like(K0)
    K0_inv[:, 0, 0] =  1.0 / fx0
    K0_inv[:, 1, 1] =  1.0 / fy0
    K0_inv[:, 0, 2] = -cx0 / fx0
    K0_inv[:, 1, 2] = -cy0 / fy0
    K0_inv[:, 2, 2] =  1.0

    ones  = torch.ones_like(track0[..., :1])
    uvh   = torch.cat([track0, ones], dim=-1)           # B, N, 3
    rays  = torch.bmm(uvh, K0_inv.transpose(-1, -2))    # B, N, 3  (cam-space rays)
    cam_pts_0 = rays * sampled_depth.unsqueeze(-1)       # B, N, 3

    E0 = pred_extrinsics[:, 0, :, :]   # B, 3, 4
    R0 = E0[:, :, :3]                  # B, 3, 3
    t0 = E0[:, :, 3]                   # B, 3

    # P_world = R0^T @ (P_cam - t0)
    world_pts = torch.bmm(cam_pts_0 - t0.unsqueeze(1), R0)   # B, N, 3
    # Clamp world points — if camera predictions are bad (large t0), world_pts
    # can overflow bfloat16 bmm in the reprojection loop → Inf*0=NaN backward.
    world_pts = world_pts.clamp(min=-1e3, max=1e3)

    # --- Reproject to all S frames ---
    total_loss = 0.0
    n_valid    = 0

    for j in range(S):
        valid_j = first_vis & gt_vis[:, j, :]   # B, N
        if not valid_j.any():
            continue

        Ej = pred_extrinsics[:, j, :, :]   # B, 3, 4
        Rj = Ej[:, :, :3]                  # B, 3, 3
        tj = Ej[:, :, 3]                   # B, 3
        Kj = pred_intrinsics[:, j, :, :]   # B, 3, 3

        # P_cam_j = R_j @ P_world + t_j
        cam_pts_j = torch.bmm(world_pts, Rj.transpose(-1, -2)) + tj.unsqueeze(1)  # B, N, 3
        cam_pts_j = cam_pts_j.clamp(min=-1e3, max=1e3)  # prevent overflow from large tj

        # Project: [u, v] = K_j @ P_cam_j / z
        proj = torch.bmm(cam_pts_j, Kj.transpose(-1, -2))   # B, N, 3
        z    = proj[..., 2:3].clamp(min=1e-3)
        uv_pred = proj[..., :2] / z   # B, N, 2

        err = (uv_pred - gt_tracks[:, j, :, :]).abs().mean(dim=-1)   # B, N
        err = check_and_fix_inf_nan(err, f"reproj_err_j{j}", hard_max=None)
        err = err.clamp(max=100.0)

        total_loss = total_loss + err[valid_j].mean()
        n_valid    += 1

    if n_valid == 0:
        return None

    return total_loss / n_valid


def _render_plane_fitting_figure(world_pts_f32, normals, offsets):
    """
    Render a 3D scatter plot of unprojected world points per frame and the
    mean fitted plane as a semi-transparent surface.

    Uses the first batch element only. Intended for TensorBoard image logging.

    Args:
        world_pts_f32 : (B, S, N, 3) float32 world-space points
        normals       : (B, S, 3)    per-frame plane normals (sign-aligned)
        offsets       : (B, S)       per-frame plane offsets

    Returns:
        (3, H, W) uint8 tensor suitable for tb_writer.log_visuals
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    pts   = world_pts_f32[0].detach().cpu().numpy()   # S, N, 3
    norms = normals[0].detach().cpu().numpy()          # S, 3
    offs  = offsets[0].detach().cpu().numpy()          # S

    S = pts.shape[0]
    fig = plt.figure(figsize=(8, 6))
    ax  = fig.add_subplot(111, projection='3d')

    colors = plt.cm.rainbow(np.linspace(0, 1, S))
    for i in range(S):
        ax.scatter(pts[i, :, 0], pts[i, :, 1], pts[i, :, 2],
                   c=[colors[i]], alpha=0.4, s=4, label=f'f{i}')

    # Mean fitted plane
    n_mean = norms.mean(axis=0)
    n_norm = np.linalg.norm(n_mean)
    if n_norm > 1e-6:
        n_mean /= n_norm
    centroid = pts.reshape(-1, 3).mean(axis=0)

    # Two orthogonal vectors spanning the plane
    u  = np.array([1.0, 0.0, 0.0]) if abs(n_mean[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    v1 = np.cross(n_mean, u);  v1 /= (np.linalg.norm(v1) + 1e-8)
    v2 = np.cross(n_mean, v1); v2 /= (np.linalg.norm(v2) + 1e-8)

    extent = float(np.ptp(pts.reshape(-1, 3), axis=0).max()) * 0.5
    g = np.linspace(-extent, extent, 12)
    g1, g2 = np.meshgrid(g, g)
    plane   = centroid + g1[..., None] * v1 + g2[..., None] * v2
    ax.plot_surface(plane[..., 0], plane[..., 1], plane[..., 2],
                    alpha=0.15, color='gray')

    # Normal arrow from centroid
    ax.quiver(*centroid, *(n_mean * extent * 0.4),
              color='red', linewidth=2, arrow_length_ratio=0.2)

    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    if S <= 16:
        ax.legend(fontsize=5, loc='upper right', markerscale=2)
    ax.set_title('Plane fitting — world pts per frame', fontsize=9)
    plt.tight_layout()

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
    plt.close(fig)

    return torch.from_numpy(img.copy()).permute(2, 0, 1)  # 3, H, W  uint8


def _fit_plane_svd(pts):
    """
    Fit a plane to a batch of 3D points using SVD.

    Args:
        pts : (B, N, 3) float32 — world-space points

    Returns:
        normal : (B, 3) unit normal vector (direction of smallest variance)
        offset : (B,)  plane offset d such that n · x = d for points on the plane
    """
    centroid = pts.mean(dim=1, keepdim=True)              # B, 1, 3
    centered = pts - centroid                              # B, N, 3
    # SVD: centered = U @ diag(S) @ Vh
    # The normal is the right singular vector for the smallest singular value (last row of Vh)
    _, _, Vh = torch.linalg.svd(centered, full_matrices=False)  # Vh: B, 3, 3
    normal = Vh[:, 2, :]                                   # B, 3
    normal = F.normalize(normal, dim=-1)
    offset = (centroid.squeeze(1) * normal).sum(dim=-1)    # B
    return normal, offset


def compute_plane_rigidity_loss(
    predictions,
    batch,
    weight=1.0,
    consistency_weight=1.0,
    plane_weight=0.0,
    min_visible_frames=2,
    visualize=False,
    **kwargs,
):
    """
    Self-supervised cross-frame 3D consistency loss for rigid plane tracks.

    Two complementary terms, each independently weighted:

    1. Point consistency (consistency_weight):
       Per-track: the L2 distance of each frame's unprojected 3D position from
       the cross-frame mean. Zero iff every (depth, camera) pair agrees on the
       same 3D world point for each track.

    2. Plane fitting (plane_weight):
       For each frame, fit a plane to all N unprojected world points via SVD.
       Loss = normal-direction variance + offset variance across frames.
       Captures systematic errors (wrong rotation, wrong depth scale) that the
       per-point term may miss. SVD computed in float32 for numerical stability.

    Gradient flows jointly through depth head and camera head — no external labels.

    Args:
        predictions        : dict with "pose_enc_list" (list of B,S,9) and "depth" (B,S,H,W,1)
        batch              : dict with "tracks" (B,S,N,2) and "track_vis_mask" (B,S,N bool)
        weight             : overall scalar weight (absorbed by caller)
        consistency_weight : weight for the point consistency term (0.0 to disable)
        plane_weight       : weight for the plane fitting term (0.0 to disable)
        min_visible_frames : track must be visible in ≥ this many frames to contribute
        visualize          : if True and plane_weight > 0, also return a (3,H,W) uint8
                             TensorBoard image of the fitted plane. Adds matplotlib
                             overhead — only set True at logging frequency.

    Returns:
        loss scalar (or None if no valid tracks) when visualize=False.
        (loss, vis_image | None) tuple when visualize=True.
    """
    if "tracks" not in batch or batch["tracks"] is None:
        return None

    gt_tracks = batch["tracks"]          # B, S, N, 2
    gt_vis    = batch["track_vis_mask"]  # B, S, N  (bool)
    _, S, _, _ = gt_tracks.shape

    # Only supervise tracks visible in at least min_visible_frames frames
    n_vis = gt_vis.sum(dim=1)                          # B, N
    valid_tracks = n_vis >= min_visible_frames          # B, N
    if not valid_tracks.any():
        return None

    H, W = batch["images"].shape[-2:]

    # Decode last-stage pose encoding → extrinsics (B,S,3,4), intrinsics (B,S,3,3)
    pred_pose_enc = predictions["pose_enc_list"][-1]
    pred_extrinsics, pred_intrinsics = pose_encoding_to_extri_intri(pred_pose_enc, (H, W))

    # Unproject GT tracks in every frame independently → world-space 3D points
    world_pts_list = []  # will be S tensors of shape (B, N, 3)

    for i in range(S):
        pred_depth_i = predictions["depth"][:, i, :, :, 0]  # B, H, W
        track_i      = gt_tracks[:, i, :, :]                 # B, N, 2

        # For invisible tracks in frame i, clamp coordinates to image bounds so
        # grid_sample doesn't receive out-of-range values and return zero depth,
        # which would produce garbage cam_pts. The resulting world_i for invisible
        # tracks is excluded from the loss via loss_mask anyway.
        vis_i  = gt_vis[:, i, :]                             # B, N
        safe_u = track_i[..., 0].clamp(0, W - 1)
        safe_v = track_i[..., 1].clamp(0, H - 1)
        safe_u = torch.where(vis_i, track_i[..., 0], safe_u)
        safe_v = torch.where(vis_i, track_i[..., 1], safe_v)

        # Bilinear sample of predicted depth at track positions
        norm_u = (safe_u / (W - 1)) * 2.0 - 1.0             # B, N
        norm_v = (safe_v / (H - 1)) * 2.0 - 1.0             # B, N
        grid   = torch.stack([norm_u, norm_v], dim=-1).unsqueeze(1)  # B, 1, N, 2
        depth_i = F.grid_sample(
            pred_depth_i.unsqueeze(1),
            grid, mode="bilinear", align_corners=True,
        ).squeeze(1).squeeze(1).clamp(min=1e-3, max=1e3)    # B, N

        # Unproject to camera space: P_cam = K^{-1} [u,v,1] * d
        Ki  = pred_intrinsics[:, i, :, :]                    # B, 3, 3
        fx  = Ki[:, 0, 0].clamp(min=1e-2)                   # B
        fy  = Ki[:, 1, 1].clamp(min=1e-2)                   # B
        cx  = Ki[:, 0, 2]                                    # B
        cy  = Ki[:, 1, 2]                                    # B
        x_c = (safe_u - cx.unsqueeze(1)) / fx.unsqueeze(1) * depth_i  # B, N
        y_c = (safe_v - cy.unsqueeze(1)) / fy.unsqueeze(1) * depth_i  # B, N
        cam_pts_i = torch.stack([x_c, y_c, depth_i], dim=-1)          # B, N, 3

        # Transform to world space: P_world = R^T @ (P_cam - t)
        Ei = pred_extrinsics[:, i, :, :]                     # B, 3, 4
        Ri = Ei[:, :, :3]                                    # B, 3, 3
        ti = Ei[:, :, 3]                                     # B, 3
        world_i = torch.bmm(cam_pts_i - ti.unsqueeze(1), Ri) # B, N, 3
        world_i = world_i.clamp(min=-1e3, max=1e3)

        world_pts_list.append(world_i)

    # Stack across frames: (B, S, N, 3)
    world_pts = torch.stack(world_pts_list, dim=1)

    # Mask: frame must be visible AND track must appear in enough frames
    loss_mask = gt_vis & valid_tracks.unsqueeze(1)           # B, S, N
    if not loss_mask.any():
        return None

    total_loss = world_pts.new_zeros(1).squeeze()

    # -------------------------------------------------------------------------
    # Term 1 — Point consistency
    # Per-track L2 distance from the cross-frame mean world position.
    # -------------------------------------------------------------------------
    if consistency_weight > 0.0:
        vis_f      = gt_vis.unsqueeze(-1).float()            # B, S, N, 1
        vis_sum    = vis_f.sum(dim=1).clamp(min=1.0)         # B, N, 1
        world_mean = (world_pts * vis_f).sum(dim=1) / vis_sum  # B, N, 3

        diff    = world_pts - world_mean.unsqueeze(1)        # B, S, N, 3
        l2_dist = (diff * diff).sum(dim=-1).add(1e-6).sqrt()  # B, S, N  — eps prevents 1/(2√x)→∞ in backward

        consistency_loss = l2_dist[loss_mask].mean()
        consistency_loss = check_and_fix_inf_nan(
            consistency_loss, "plane_consistency_loss", hard_max=100.0
        )
        total_loss = total_loss + consistency_weight * consistency_loss

    # -------------------------------------------------------------------------
    # Term 2 — Plane fitting
    # For each frame, fit a plane via SVD to its N world points.
    # Loss = variance of plane normals + variance of plane offsets across frames.
    # Captures systematic errors (wrong rotation, wrong depth scale) that the
    # per-point term may miss.
    # SVD is computed in float32 — bfloat16 SVD backward is numerically unstable.
    # -------------------------------------------------------------------------
    if plane_weight > 0.0:
        world_pts_f32 = world_pts.float()                    # B, S, N, 3 in float32

        normals_list = []
        offsets_list = []
        for i in range(S):
            n_i, d_i = _fit_plane_svd(world_pts_f32[:, i, :, :])  # B,3 and B
            normals_list.append(n_i)
            offsets_list.append(d_i)

        normals = torch.stack(normals_list, dim=1)           # B, S, 3
        offsets = torch.stack(offsets_list, dim=1)           # B, S

        # Resolve sign ambiguity: flip normals that point opposite to frame 0.
        # This ensures all normals describe the same side of the plane before
        # computing variance.
        dot     = (normals * normals[:, :1, :]).sum(dim=-1)  # B, S
        signs   = dot.sign()
        signs   = torch.where(signs == 0, torch.ones_like(signs), signs)
        normals = normals * signs.unsqueeze(-1)              # B, S, 3
        offsets = offsets * signs                            # B, S

        # Normal variance: 1 - cos^2(angle to mean normal).
        # Zero when all normals are identical; max 1 when orthogonal.
        n_mean        = F.normalize(normals.mean(dim=1), dim=-1)  # B, 3
        cos_sim       = (normals * n_mean.unsqueeze(1)).sum(dim=-1)  # B, S
        normal_var    = (1.0 - cos_sim.pow(2)).mean()

        # Offset variance: mean squared deviation from the mean offset.
        d_mean        = offsets.mean(dim=1, keepdim=True)    # B, 1
        offset_var    = (offsets - d_mean).pow(2).mean()

        plane_loss = normal_var + offset_var
        plane_loss = check_and_fix_inf_nan(
            plane_loss.to(world_pts.dtype), "plane_fitting_loss", hard_max=100.0
        )
        total_loss = total_loss + plane_weight * plane_loss

    else:
        world_pts_f32 = None
        normals = None
        offsets = None

    total_loss = check_and_fix_inf_nan(total_loss, "plane_rigidity_loss", hard_max=100.0)

    if visualize:
        if world_pts_f32 is None:  # plane_weight=0 — compute SVD for viz only, no grad
            with torch.no_grad():
                world_pts_f32 = world_pts.float()
                normals_list, offsets_list = [], []
                for i in range(S):
                    n_i, d_i = _fit_plane_svd(world_pts_f32[:, i, :, :])
                    normals_list.append(n_i)
                    offsets_list.append(d_i)
                normals = torch.stack(normals_list, dim=1)
                offsets = torch.stack(offsets_list, dim=1)
                dot    = (normals * normals[:, :1, :]).sum(dim=-1)
                signs  = dot.sign()
                signs  = torch.where(signs == 0, torch.ones_like(signs), signs)
                normals = normals * signs.unsqueeze(-1)
                offsets = offsets * signs
        vis_image = _render_plane_fitting_figure(world_pts_f32, normals, offsets)
        return total_loss, vis_image
    return total_loss


def reduce_masked_mean(x, mask, dim=None, keepdim=False):
    for a, b in zip(x.size(), mask.size()):
        assert a == b
    prod = x * mask

    if dim is None:
        numer = torch.sum(prod)
        denom = torch.sum(mask)
    else:
        numer = torch.sum(prod, dim=dim, keepdim=keepdim)
        denom = torch.sum(mask, dim=dim, keepdim=keepdim)

    mean = numer / denom.clamp(min=1)
    mean = torch.where(denom > 0, mean, torch.zeros_like(mean))
    return mean


def sequence_loss(flow_preds, flow_gt, vis, valids, gamma=0.8, vis_aware=False, vis_aware_w=0.1, **kwargs):
    """Loss function defined over sequence of flow predictions."""
    B, S, N, D = flow_gt.shape
    assert D == 2
    assert vis.shape == (B, S, N)
    assert valids.shape == (B, S, N)

    n_predictions = len(flow_preds)
    flow_loss = 0.0

    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()          # B, S, N, 2
        i_loss = check_and_fix_inf_nan(i_loss, f"i_loss_iter_{i}", hard_max=None)
        i_loss = torch.mean(i_loss, dim=3)                 # B, S, N

        combined_mask = torch.logical_and(valids, vis)
        num_valid = combined_mask.sum()

        if vis_aware:
            combined_mask = combined_mask.float() * (1.0 + vis_aware_w)
            flow_loss += i_weight * reduce_masked_mean(i_loss, combined_mask)
        else:
            if num_valid > 2:
                flow_loss += i_weight * i_loss[combined_mask].mean()
            else:
                i_loss = check_and_fix_inf_nan(i_loss, f"i_loss_safe_{i}", hard_max=None)
                flow_loss += 0 * i_loss.mean()

    if n_predictions > 0:
        flow_loss = flow_loss / n_predictions

    return flow_loss
