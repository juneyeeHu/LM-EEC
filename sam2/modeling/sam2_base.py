# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.distributed
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.init import trunc_normal_

from sam2.modeling.sam.mask_decoder import MaskDecoder
from sam2.modeling.sam.prompt_encoder import PromptEncoder, ViewPromptEncoder
from sam2.modeling.sam.transformer import TwoWayTransformer
from sam2.modeling.sam2_utils import get_1d_sine_pe, MLP, select_closest_cond_frames
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
# a large negative value as a placeholder score for missing objects
NO_OBJ_SCORE = -1024.0


class SAM2Base(torch.nn.Module):
    def __init__(
        self,
        image_encoder,
        memory_attention,
        memory_encoder,
        num_maskmem=7,  # default 1 input frame + 6 previous frames
        image_size=512,
        backbone_stride=16,  # stride of the image backbone output
        sigmoid_scale_for_mem_enc=1.0,  # scale factor for mask sigmoid prob
        sigmoid_bias_for_mem_enc=0.0,  # bias factor for mask sigmoid prob
        # During evaluation, whether to binarize the sigmoid mask logits on interacted frames with clicks
        binarize_mask_from_pts_for_mem_enc=False,
        use_mask_input_as_output_without_sam=False,  # on frames with mask input, whether to directly output the input mask without using a SAM prompt encoder + mask decoder
        # The maximum number of conditioning frames to participate in the memory attention (-1 means no limit; if there are more conditioning frames than this limit,
        # we only cross-attend to the temporally closest `max_cond_frames_in_attn` conditioning frames in the encoder when tracking each frame). This gives the model
        # a temporal locality when handling a large number of annotated frames (since closer frames should be more important) and also avoids GPU OOM.
        max_cond_frames_in_attn=-1,
        # on the first frame, whether to directly add the no-memory embedding to the image feature
        # (instead of using the transformer encoder)
        directly_add_no_mem_embed=False,
        # whether to use high-resolution feature maps in the SAM mask decoder
        use_high_res_features_in_sam=False,
        # whether to output multiple (3) masks for the first click on initial conditioning frames
        multimask_output_in_sam=False,
        # the minimum and maximum number of clicks to use multimask_output_in_sam (only relevant when `multimask_output_in_sam=True`;
        # default is 1 for both, meaning that only the first click gives multimask output; also note that a box counts as two points)
        multimask_min_pt_num=1,
        multimask_max_pt_num=1,
        # whether to also use multimask output for tracking (not just for the first click on initial conditioning frames; only relevant when `multimask_output_in_sam=True`)
        multimask_output_for_tracking=False,
        # Whether to use multimask tokens for obj ptr; Only relevant when both
        # use_obj_ptrs_in_encoder=True and multimask_output_for_tracking=True
        use_multimask_token_for_obj_ptr: bool = False,
        # whether to use sigmoid to restrict ious prediction to [0-1]
        iou_prediction_use_sigmoid=False,
        # The memory bank's temporal stride during evaluation (i.e. the `r` parameter in XMem and Cutie; XMem and Cutie use r=5).
        # For r>1, the (self.num_maskmem - 1) non-conditioning memory frames consist of
        # (self.num_maskmem - 2) nearest frames from every r-th frames, plus the last frame.
        memory_temporal_stride_for_eval=1,
        # whether to apply non-overlapping constraints on the object masks in the memory encoder during evaluation (to avoid/alleviate superposing masks)
        non_overlap_masks_for_mem_enc=False,
        # whether to cross-attend to object pointers from other frames (based on SAM output tokens) in the encoder
        use_obj_ptrs_in_encoder=False,
        # the maximum number of object pointers from other frames in encoder cross attention (only relevant when `use_obj_ptrs_in_encoder=True`)
        max_obj_ptrs_in_encoder=16,
        # whether to add temporal positional encoding to the object pointers in the encoder (only relevant when `use_obj_ptrs_in_encoder=True`)
        add_tpos_enc_to_obj_ptrs=True,
        # whether to add an extra linear projection layer for the temporal positional encoding in the object pointers to avoid potential interference
        # with spatial positional encoding (only relevant when both `use_obj_ptrs_in_encoder=True` and `add_tpos_enc_to_obj_ptrs=True`)
        proj_tpos_enc_in_obj_ptrs=False,
        # whether to use signed distance (instead of unsigned absolute distance) in the temporal positional encoding in the object pointers
        # (only relevant when both `use_obj_ptrs_in_encoder=True` and `add_tpos_enc_to_obj_ptrs=True`)
        use_signed_tpos_enc_to_obj_ptrs=False,
        # whether to only attend to object pointers in the past (before the current frame) in the encoder during evaluation
        # (only relevant when `use_obj_ptrs_in_encoder=True`; this might avoid pointer information too far in the future to distract the initial tracking)
        only_obj_ptrs_in_the_past_for_eval=False,
        # Whether to predict if there is an object in the frame
        pred_obj_scores: bool = False,
        # Whether to use an MLP to predict object scores
        pred_obj_scores_mlp: bool = False,
        # Only relevant if pred_obj_scores=True and use_obj_ptrs_in_encoder=True;
        # Whether to have a fixed no obj pointer when there is no object present
        # or to use it as an additive embedding with obj_ptr produced by decoder
        fixed_no_obj_ptr: bool = False,
        # Soft no object, i.e. mix in no_obj_ptr softly,
        # hope to make recovery easier if there is a mistake and mitigate accumulation of errors
        soft_no_obj_ptr: bool = False,
        use_mlp_for_obj_ptr_proj: bool = False,
        # add no obj embedding to spatial frames
        no_obj_embed_spatial: bool = False,
        # extra arguments used to construct the SAM mask decoder; if not None, it should be a dict of kwargs to be passed into `MaskDecoder` class.
        sam_mask_decoder_extra_args=None,
        compile_image_encoder: bool = False,
    ):
        super().__init__()

        # Part 1: the image backbone
        self.image_encoder = image_encoder
        # Use level 0, 1, 2 for high-res setting, or just level 2 for the default setting
        self.use_high_res_features_in_sam = use_high_res_features_in_sam
        self.num_feature_levels = 3 if use_high_res_features_in_sam else 1
        self.use_obj_ptrs_in_encoder = use_obj_ptrs_in_encoder
        self.max_obj_ptrs_in_encoder = max_obj_ptrs_in_encoder
        if use_obj_ptrs_in_encoder:
            # A conv layer to downsample the mask prompt to stride 4 (the same stride as
            # low-res SAM mask logits) and to change its scales from 0~1 to SAM logit scale,
            # so that it can be fed into the SAM mask decoder to generate a pointer.
            self.mask_downsample = torch.nn.Conv2d(1, 1, kernel_size=4, stride=4)
        self.add_tpos_enc_to_obj_ptrs = add_tpos_enc_to_obj_ptrs
        if proj_tpos_enc_in_obj_ptrs:
            assert add_tpos_enc_to_obj_ptrs  # these options need to be used together
        self.proj_tpos_enc_in_obj_ptrs = proj_tpos_enc_in_obj_ptrs
        self.use_signed_tpos_enc_to_obj_ptrs = use_signed_tpos_enc_to_obj_ptrs
        self.only_obj_ptrs_in_the_past_for_eval = only_obj_ptrs_in_the_past_for_eval
        # self.toPrototype = P2SP(64, [3, 3])
        # Part 2: memory attention to condition current frame's visual features
        # with memories (and obj ptrs) from past frames
        self.memory_attention = memory_attention
        self.hidden_dim = image_encoder.neck.d_model

        # Part 3: memory encoder for the previous frame's outputs
        self.memory_encoder = memory_encoder
        self.mem_dim = self.hidden_dim
        if hasattr(self.memory_encoder, "out_proj") and hasattr(
            self.memory_encoder.out_proj, "weight"
        ):
            # if there is compression of memories along channel dim
            self.mem_dim = self.memory_encoder.out_proj.weight.shape[0]
        self.num_maskmem = num_maskmem  # Number of memories accessible
        # Temporal encoding of the memories
        self.maskmem_tpos_enc = torch.nn.Parameter(
            torch.zeros(num_maskmem, 1, 1, self.mem_dim)
        )
        trunc_normal_(self.maskmem_tpos_enc, std=0.02)
        # a single token to indicate no memory embedding from previous frames
        self.no_mem_embed = torch.nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.no_mem_pos_enc = torch.nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        trunc_normal_(self.no_mem_embed, std=0.02)
        trunc_normal_(self.no_mem_pos_enc, std=0.02)
        self.directly_add_no_mem_embed = directly_add_no_mem_embed
        # Apply sigmoid to the output raw mask logits (to turn them from
        # range (-inf, +inf) to range (0, 1)) before feeding them into the memory encoder
        self.sigmoid_scale_for_mem_enc = sigmoid_scale_for_mem_enc
        self.sigmoid_bias_for_mem_enc = sigmoid_bias_for_mem_enc
        self.binarize_mask_from_pts_for_mem_enc = binarize_mask_from_pts_for_mem_enc
        self.non_overlap_masks_for_mem_enc = non_overlap_masks_for_mem_enc
        self.memory_temporal_stride_for_eval = memory_temporal_stride_for_eval
        # On frames with mask input, whether to directly output the input mask without
        # using a SAM prompt encoder + mask decoder
        self.use_mask_input_as_output_without_sam = use_mask_input_as_output_without_sam
        self.multimask_output_in_sam = multimask_output_in_sam
        self.multimask_min_pt_num = multimask_min_pt_num
        self.multimask_max_pt_num = multimask_max_pt_num
        self.multimask_output_for_tracking = multimask_output_for_tracking
        self.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr
        self.iou_prediction_use_sigmoid = iou_prediction_use_sigmoid

        # Part 4: SAM-style prompt encoder (for both mask and point inputs)
        # and SAM-style mask decoder for the final mask output
        self.image_size = image_size
        self.backbone_stride = backbone_stride
        self.sam_mask_decoder_extra_args = sam_mask_decoder_extra_args
        self.pred_obj_scores = pred_obj_scores
        self.pred_obj_scores_mlp = pred_obj_scores_mlp
        self.fixed_no_obj_ptr = fixed_no_obj_ptr
        self.soft_no_obj_ptr = soft_no_obj_ptr
        if self.fixed_no_obj_ptr:
            assert self.pred_obj_scores
            assert self.use_obj_ptrs_in_encoder
        if self.pred_obj_scores and self.use_obj_ptrs_in_encoder:
            self.no_obj_ptr = torch.nn.Parameter(torch.zeros(1, self.hidden_dim))
            trunc_normal_(self.no_obj_ptr, std=0.02)
        self.use_mlp_for_obj_ptr_proj = use_mlp_for_obj_ptr_proj
        self.no_obj_embed_spatial = None
        if no_obj_embed_spatial:
            self.no_obj_embed_spatial = torch.nn.Parameter(torch.zeros(1, self.mem_dim))
            trunc_normal_(self.no_obj_embed_spatial, std=0.02)

        self._build_sam_heads()
        self.max_cond_frames_in_attn = max_cond_frames_in_attn
        self.MV_MoE = MV_MoE(256)
        # Model compilation
        if compile_image_encoder:
            # Compile the forward function (not the full module) to allow loading checkpoints.
            print(
                "Image encoder compilation is enabled. First forward pass will be slow."
            )
            self.image_encoder.forward = torch.compile(
                self.image_encoder.forward,
                mode="max-autotune",
                fullgraph=True,
                dynamic=False,
            )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "Please use the corresponding methods in SAM2VideoPredictor for inference or SAM2Train for training/fine-tuning"
            "See notebooks/video_predictor_example.ipynb for an inference example."
        )


    def _build_sam_heads(self):
        """Build SAM-style prompt encoder and mask decoder."""
        self.sam_prompt_embed_dim = self.hidden_dim
        self.sam_image_embedding_size = self.image_size // self.backbone_stride
        # build PromptEncoder and MaskDecoder from SAM
        # (their hyperparameters like `mask_in_chans=16` are from SAM code)
        self.sam_prompt_encoder = PromptEncoder(
            embed_dim=self.sam_prompt_embed_dim,
            image_embedding_size=(
                self.sam_image_embedding_size,
                self.sam_image_embedding_size,
            ),
            input_image_size=(self.image_size, self.image_size),
            mask_in_chans=16,
        )
        self.sam_mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=self.sam_prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=self.sam_prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            use_high_res_features=self.use_high_res_features_in_sam,
            iou_prediction_use_sigmoid=self.iou_prediction_use_sigmoid,
            pred_obj_scores=self.pred_obj_scores,
            pred_obj_scores_mlp=self.pred_obj_scores_mlp,
            use_multimask_token_for_obj_ptr=self.use_multimask_token_for_obj_ptr,
            **(self.sam_mask_decoder_extra_args or {}),
        )
        if self.use_obj_ptrs_in_encoder:
            # a linear projection on SAM output tokens to turn them into object pointers
            self.obj_ptr_proj = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
            if self.use_mlp_for_obj_ptr_proj:
                self.obj_ptr_proj = MLP(
                    self.hidden_dim, self.hidden_dim, self.hidden_dim, 3
                )
        else:
            self.obj_ptr_proj = torch.nn.Identity()
        if self.proj_tpos_enc_in_obj_ptrs:
            # a linear projection on temporal positional encoding in object pointers to
            # avoid potential interference with spatial positional encoding
            self.obj_ptr_tpos_proj = torch.nn.Linear(self.hidden_dim, self.mem_dim)
        else:
            self.obj_ptr_tpos_proj = torch.nn.Identity()

    def _forward_sam_heads(
        self,
        ego_backbone_features,
        exo_backbone_features,
        exo_point_inputs=None,
        ego_mask_inputs=None,
        exo_mask_inputs=None,
        ego_high_res_features=None,
        exo_high_res_features=None,
        multimask_output=False,
        ego_gt_masks=None,
        exo_gt_masks = None
    ):
        """
        Forward SAM prompt encoders and mask heads.

        Inputs:
        - backbone_features: image features of [B, C, H, W] shape
        - point_inputs: a dictionary with "point_coords" and "point_labels", where
          1) "point_coords" has [B, P, 2] shape and float32 dtype and contains the
             absolute pixel-unit coordinate in (x, y) format of the P input points
          2) "point_labels" has shape [B, P] and int32 dtype, where 1 means
             positive clicks, 0 means negative clicks, and -1 means padding
        - mask_inputs: a mask of [B, 1, H*16, W*16] shape, float or bool, with the
          same spatial size as the image.
        - high_res_features: either 1) None or 2) or a list of length 2 containing
          two feature maps of [B, C, 4*H, 4*W] and [B, C, 2*H, 2*W] shapes respectively,
          which will be used as high-resolution feature maps for SAM decoder.
        - multimask_output: if it's True, we output 3 candidate masks and their 3
          corresponding IoU estimates, and if it's False, we output only 1 mask and
          its corresponding IoU estimate.

        Outputs:
        - low_res_multimasks: [B, M, H*4, W*4] shape (where M = 3 if
          `multimask_output=True` and M = 1 if `multimask_output=False`), the SAM
          output mask logits (before sigmoid) for the low-resolution masks, with 4x
          the resolution (1/4 stride) of the input backbone_features.
        - high_res_multimasks: [B, M, H*16, W*16] shape (where M = 3
          if `multimask_output=True` and M = 1 if `multimask_output=False`),
          upsampled from the low-resolution masks, with shape size as the image
          (stride is 1 pixel).
        - ious, [B, M] shape, where (where M = 3 if `multimask_output=True` and M = 1
          if `multimask_output=False`), the estimated IoU of each output mask.
        - low_res_masks: [B, 1, H*4, W*4] shape, the best mask in `low_res_multimasks`.
          If `multimask_output=True`, it's the mask with the highest IoU estimate.
          If `multimask_output=False`, it's the same as `low_res_multimasks`.
        - high_res_masks: [B, 1, H*16, W*16] shape, the best mask in `high_res_multimasks`.
          If `multimask_output=True`, it's the mask with the highest IoU estimate.
          If `multimask_output=False`, it's the same as `high_res_multimasks`.
        - obj_ptr: [B, C] shape, the object pointer vector for the output mask, extracted
          based on the output token from the SAM mask decoder.
        """
        B = exo_backbone_features.size(0) # B=1 如果是batchsize为2这里的值为2吗？
        # print("B:", B)
        device = exo_backbone_features.device
        assert exo_backbone_features.size(1) == self.sam_prompt_embed_dim
        assert exo_backbone_features.size(2) == self.sam_image_embedding_size
        assert exo_backbone_features.size(3) == self.sam_image_embedding_size

        # a) Handle point prompts
        if exo_point_inputs is not None:
            sam_point_coords = exo_point_inputs["point_coords"]
            sam_point_labels = exo_point_inputs["point_labels"]
            assert sam_point_coords.size(0) == B and sam_point_labels.size(0) == B
        else:
            # If no points are provide, pad with an empty point (with label -1)
            sam_point_coords = torch.zeros(B, 1, 2, device=device)
            sam_point_labels = -torch.ones(B, 1, dtype=torch.int32, device=device)

        # b) Handle mask prompts
        if ego_gt_masks is not None:
            # If mask_inputs is provided, downsize it into low-res mask input if needed
            # and feed it as a dense mask prompt into the SAM mask encoder
            assert len(ego_gt_masks.shape) == 4 and ego_gt_masks.shape[:2] == (B, 1)
            if ego_gt_masks.shape[-2:] != self.sam_prompt_encoder.mask_input_size:
                sam_mask_prompt = F.interpolate(
                    ego_gt_masks.float(),
                    size=self.sam_prompt_encoder.mask_input_size,
                    align_corners=False,
                    mode="bilinear",
                    antialias=True,  # use antialias for downsampling
                )
            else:
                sam_mask_prompt = ego_gt_masks
        else:
            # Otherwise, simply feed None (and SAM's prompt encoder will add
            # a learned `no_mask_embed` to indicate no mask input in this case).
            sam_mask_prompt = None

        if exo_gt_masks is not None:
            assert len(exo_gt_masks.shape) == 4 and exo_gt_masks.shape[:2] == (B, 1)
            if exo_gt_masks.shape[-2:] != self.sam_prompt_encoder.mask_input_size:
                sam_mask_prompt_exo = F.interpolate(
                    exo_gt_masks.float(),
                    size=self.sam_prompt_encoder.mask_input_size,
                    align_corners=False,
                    mode="bilinear",
                    antialias=True,  # use antialias for downsampling
                )
            else:
                sam_mask_prompt_exo = exo_gt_masks
        else:
            # Otherwise, simply feed None (and SAM's prompt encoder will add
            # a learned `no_mask_embed` to indicate no mask input in this case).
            sam_mask_prompt_exo = None
        sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(
            points=(sam_point_coords, sam_point_labels),
            boxes=None,
            masks=sam_mask_prompt,
        )

        (
            low_res_multimasks, # 【1，1，256，256】
            ious,
            sam_output_tokens,
            object_score_logits,
            upscaled_embedding,
            src,
            res
        ) = self.sam_mask_decoder(
            image_embeddings=exo_backbone_features,
            image_pe=self.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,  # False
            repeat_image=False,  # the image is already batched
            high_res_features=exo_high_res_features,
        )
        # teacher_feature = exo_backbone_features+exo_dense_embeddings
        if self.pred_obj_scores:
            is_obj_appearing = object_score_logits > 0

            # Mask used for spatial memories is always a *hard* choice between obj and no obj,
            # consistent with the actual mask prediction
            low_res_multimasks = torch.where(
                is_obj_appearing[:, None, None],
                low_res_multimasks,
                NO_OBJ_SCORE,
            )

        # convert masks from possibly bfloat16 (or float16) to float32
        # (older PyTorch versions before 2.1 don't support `interpolate` on bf16)
        low_res_multimasks = low_res_multimasks.float()
        high_res_multimasks = F.interpolate(
            low_res_multimasks,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        if res is not None:
            high_res_affinity = F.interpolate(
                res,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        else:
            high_res_affinity = None
        sam_output_token = sam_output_tokens[:, 0]
        if multimask_output:
            # take the best mask prediction (with the highest IoU estimation)
            best_iou_inds = torch.argmax(ious, dim=-1)
            batch_inds = torch.arange(B, device=device)
            low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            if sam_output_tokens.size(1) > 1:
                sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
            ious = ious.max(-1)[0]
        else:
            low_res_masks, high_res_masks = low_res_multimasks, high_res_multimasks

        # Extract object pointer from the SAM output token (with occlusion handling)
        obj_ptr = self.obj_ptr_proj(sam_output_token)
        if self.pred_obj_scores:
            # Allow *soft* no obj ptr, unlike for masks
            if self.soft_no_obj_ptr:
                lambda_is_obj_appearing = object_score_logits.sigmoid()
            else:
                lambda_is_obj_appearing = is_obj_appearing.float()

            if self.fixed_no_obj_ptr:
                obj_ptr = lambda_is_obj_appearing * obj_ptr # 变为0了
            obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr

        teacher_feature = None
        return (
            low_res_multimasks,
            high_res_multimasks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
            upscaled_embedding,
            src,
            teacher_feature,
            high_res_affinity
        )

    def _use_mask_as_output(self, backbone_features, high_res_features, mask_inputs):
        """
        Directly turn binary `mask_inputs` into a output mask logits without using SAM.
        (same input and output shapes as in _forward_sam_heads above).
        """
        # Use -10/+10 as logits for neg/pos pixels (very close to 0/1 in prob after sigmoid).
        out_scale, out_bias = 20.0, -10.0
        mask_inputs_float = mask_inputs.float()
        high_res_masks = mask_inputs_float * out_scale + out_bias
        low_res_masks = F.interpolate(
            high_res_masks,
            size=(high_res_masks.size(-2) // 4, high_res_masks.size(-1) // 4),
            align_corners=False,
            mode="bilinear",
            antialias=True,  # use antialias for downsampling
        )
        # a dummy IoU prediction of all 1's under mask input
        ious = mask_inputs.new_ones(mask_inputs.size(0), 1).float()
        if not self.use_obj_ptrs_in_encoder:
            # all zeros as a dummy object pointer (of shape [B, C])
            obj_ptr = torch.zeros(
                mask_inputs.size(0), self.hidden_dim, device=mask_inputs.device
            )
        else:
            # produce an object pointer using the SAM decoder from the mask input
            _, _, _, _, _, obj_ptr, _ = self._forward_sam_heads(
                backbone_features=backbone_features,
                mask_inputs=self.mask_downsample(mask_inputs_float),
                high_res_features=high_res_features,
            )
        # In this method, we are treating mask_input as output, e.g. using it directly to create spatial mem;
        # Below, we follow the same design axiom to use mask_input to decide if obj appears or not instead of relying
        # on the object_scores from the SAM decoder.
        is_obj_appearing = torch.any(mask_inputs.flatten(1).float() > 0.0, dim=1)
        is_obj_appearing = is_obj_appearing[..., None]
        lambda_is_obj_appearing = is_obj_appearing.float()
        object_score_logits = out_scale * lambda_is_obj_appearing + out_bias
        if self.pred_obj_scores:
            if self.fixed_no_obj_ptr:
                obj_ptr = lambda_is_obj_appearing * obj_ptr
            obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr

        return (
            low_res_masks,
            high_res_masks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        )

    def forward_image(self, ego_img_batch: torch.Tensor, exo_img_batch: torch.Tensor):
        """Get the image feature on the input batch."""

        ego_backbone_out = self.image_encoder(ego_img_batch)
        # print("ego_backbone_out", ego_backbone_out.shape)
        exo_backbone_out = self.image_encoder(exo_img_batch)

        if self.use_high_res_features_in_sam: # True
            # precompute projected level 0 and level 1 features in SAM decoder
            # to avoid running it again on every SAM click
            ego_backbone_out["backbone_fpn"][0] = self.sam_mask_decoder.conv_s0( # 可以这样用模块，是不是我的fuser模块也可以用？
                ego_backbone_out["backbone_fpn"][0]
            )  # [8, 32, 64, 64]
            ego_backbone_out["backbone_fpn"][1] = self.sam_mask_decoder.conv_s1(
                ego_backbone_out["backbone_fpn"][1]
            )  # [8, 64, 32, 32]
            exo_backbone_out["backbone_fpn"][0] = self.sam_mask_decoder.conv_s0(
                exo_backbone_out["backbone_fpn"][0]
            )  # [8, 32, 64, 64]
            exo_backbone_out["backbone_fpn"][1] = self.sam_mask_decoder.conv_s1(
                exo_backbone_out["backbone_fpn"][1]
            )  # [8, 64, 32, 32]

        return ego_backbone_out, exo_backbone_out

    def _prepare_backbone_features(self, backbone_out):
        """Prepare and flatten visual features."""
        backbone_out = backbone_out.copy()
        assert len(backbone_out["backbone_fpn"]) == len(backbone_out["vision_pos_enc"])
        assert len(backbone_out["backbone_fpn"]) >= self.num_feature_levels

        feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels :]
        # {[8, 32, 64, 64] [8, 64, 32, 32] [8, 256, 16, 16]}
        vision_pos_embeds = backbone_out["vision_pos_enc"][-self.num_feature_levels :]
        # {[8, 256, 64, 64] [8, 256, 32, 32] [8, 256, 16, 16]}
        feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]
        # [(64, 64), (32, 32), (16, 16)]
        # flatten NxCxHxW to HWxNxC
        vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps] # {list:3} tensor(H*W, B, C)
        vision_pos_embeds = [x.flatten(2).permute(2, 0, 1) for x in vision_pos_embeds]

        return backbone_out, vision_feats, vision_pos_embeds, feat_sizes


    def _prepare_memory_conditioned_features_wo_prompt(
        self,
        frame_idx,
        is_init_cond_frame,
        ego_current_vision_feats,
        exo_current_vision_feats,
        ego_current_vision_pos_embeds,
        exo_current_vision_pos_embeds,
        feat_sizes,
        ego_output_dict,
        exo_output_dict,
        num_frames,
        track_in_reverse=False,  # tracking in reverse time order (for demo usage)
        ego_gt_masks=None
    ):
        """Fuse the current frame's visual feature map with previous memory."""
        B = exo_current_vision_feats[-1].size(1)  # batch size on this frame
        C = self.hidden_dim # 256
        H, W = feat_sizes[-1]  # H/16, W/16 # top-level (lowest-resolution) feature size
        device = exo_current_vision_feats[-1].device
        # The case of `self.num_maskmem == 0` below is primarily used for reproducing SAM on images.
        # In this case, we skip the fusion with any memory.
        num_obj_ptr_tokens = 0
        tpos_sign_mul = -1 if track_in_reverse else 1
        # Step 1: condition the visual features of the current frame on previous memories
        if not is_init_cond_frame:
            # Retrieve the memories encoded with the maskmem backbone
            to_cat_memory, to_cat_memory_pos_embed = [], []
            ego_memories, exo_memories = [], []
            ego_memories_pos, exo_memories_pos = [], []
            # Add conditioning frames's output first (all cond frames have t_pos=0 for
            # when getting temporal positional embedding below)
            # assert len(exo_output_dict["cond_frame_outputs"]) > 0
            # Select a maximum number of temporally closest cond frames for cross attention
            cond_outputs = exo_output_dict["cond_frame_outputs"]
            selected_cond_outputs, unselected_cond_outputs = select_closest_cond_frames(
                frame_idx, cond_outputs, self.max_cond_frames_in_attn
            ) # {} {}

            t_pos_and_prevs = [(0, out) for out in selected_cond_outputs.values()]
            # Add last (self.num_maskmem - 1) frames before current frame for non-conditioning memory
            # the earliest one has t_pos=1 and the latest one has t_pos=self.num_maskmem-1
            # We also allow taking the memory frame non-consecutively (with stride>1), in which case
            # we take (self.num_maskmem - 2) frames among every stride-th frames plus the last frame.
            stride = 1 if self.training else self.memory_temporal_stride_for_eval
            for t_pos in range(1, self.num_maskmem):
                # [1, 2, 3, 4, 5, 6]
                t_rel = self.num_maskmem - t_pos # [6, 5, 4, 3, 2, 1]
                # how many frames before current frame
                if t_rel == 1:
                    # for t_rel == 1, we take the last frame (regardless of r)
                    if not track_in_reverse:
                        # the frame immediately before this frame (i.e. frame_idx - 1)
                        prev_frame_idx = frame_idx - t_rel
                    else:
                        # the frame immediately after this frame (i.e. frame_idx + 1)
                        prev_frame_idx = frame_idx + t_rel
                else:
                    # for t_rel >= 2, we take the memory frame from every r-th frames
                    if not track_in_reverse:
                        # first find the nearest frame among every r-th frames before this frame
                        # for r=1, this would be (frame_idx - 2)
                        prev_frame_idx = ((frame_idx - 2) // stride) * stride # [-1] [0]
                        # then seek further among every r-th frames
                        prev_frame_idx = prev_frame_idx - (t_rel - 2) * stride # [-5, -4, -3, -2, -1, 0] [-4, -3, -2, -1, 0, 1]
                    else:
                        # first find the nearest frame among every r-th frames after this frame
                        # for r=1, this would be (frame_idx + 2)
                        prev_frame_idx = -(-(frame_idx + 2) // stride) * stride
                        # then seek further among every r-th frames
                        prev_frame_idx = prev_frame_idx + (t_rel - 2) * stride
                exo_out = exo_output_dict["non_cond_frame_outputs"].get(prev_frame_idx, None)
                ego_out = ego_output_dict["non_cond_frame_outputs"].get(prev_frame_idx, None)
                t_pos_and_prevs.append((t_pos, exo_out, ego_out))

            for t_pos, prev_exo, prev_ego in t_pos_and_prevs:
                if prev_exo is None:
                    continue  # skip padding frames
                # "maskmem_features" might have been offloaded to CPU in demo use cases,
                # so we load it back to GPU (it's a no-op if it's already on GPU).
                exo_feats = prev_exo["maskmem_features"].to(device, non_blocking=True)
                ego_feats = prev_ego["maskmem_features"].to(device, non_blocking=True)
                to_cat_memory.append(exo_feats.flatten(2).permute(2, 0, 1))
                to_cat_memory.append(ego_feats.flatten(2).permute(2, 0, 1))
                ego_memories.append(ego_feats.flatten(2).permute(2, 0, 1))
                exo_memories.append(exo_feats.flatten(2).permute(2, 0, 1))
                # Spatial positional encoding (it might have been offloaded to CPU in eval)
                exo_maskmem_enc = prev_exo["maskmem_pos_enc"][-1].to(device)
                exo_maskmem_enc = exo_maskmem_enc.flatten(2).permute(2, 0, 1)
                # Temporal positional encoding
                exo_maskmem_enc = (
                    exo_maskmem_enc + self.maskmem_tpos_enc[self.num_maskmem - t_pos - 1]
                )

                ego_maskmem_enc = prev_ego["maskmem_pos_enc"][-1].to(device)
                ego_maskmem_enc = ego_maskmem_enc.flatten(2).permute(2, 0, 1)
                # Temporal positional encoding
                ego_maskmem_enc = (
                    ego_maskmem_enc + self.maskmem_tpos_enc[self.num_maskmem - t_pos - 1]
                )


                to_cat_memory_pos_embed.append(exo_maskmem_enc)
                to_cat_memory_pos_embed.append(ego_maskmem_enc)
                exo_memories_pos.append(exo_maskmem_enc)
                ego_memories_pos.append(ego_maskmem_enc)

            # Construct the list of past object pointers
            if self.use_obj_ptrs_in_encoder:
                max_obj_ptrs_in_encoder = min(num_frames, self.max_obj_ptrs_in_encoder)
                # First add those object pointers from selected conditioning frames
                # (optionally, only include object pointers in the past during evaluation)
                if not self.training and self.only_obj_ptrs_in_the_past_for_eval:
                    ptr_cond_outputs = {
                        t: out
                        for t, out in selected_cond_outputs.items()
                        if (t >= frame_idx if track_in_reverse else t <= frame_idx)
                    }
                else:
                    ptr_cond_outputs = selected_cond_outputs # {}
                pos_and_ptrs = [
                    # Temporal pos encoding contains how far away each pointer is from current frame
                    (
                        (
                            (frame_idx - t) * tpos_sign_mul
                            if self.use_signed_tpos_enc_to_obj_ptrs
                            else abs(frame_idx - t)
                        ),
                        out["obj_ptr"],
                    )
                    for t, out in ptr_cond_outputs.items()
                ]
                # Add up to (max_obj_ptrs_in_encoder - 1) non-conditioning frames before current frame
                for t_diff in range(1, max_obj_ptrs_in_encoder): # 1<= t_diff <8
                    t = frame_idx + t_diff if track_in_reverse else frame_idx - t_diff
                    if t < 0 or (num_frames is not None and t >= num_frames):
                        break
                    out = exo_output_dict["non_cond_frame_outputs"].get(
                        t, unselected_cond_outputs.get(t, None)
                    )
                    if out is not None:
                        pos_and_ptrs.append((t_diff, out["obj_ptr"]))
                # If we have at least one object pointer, add them to the across attention
                if len(pos_and_ptrs) > 0:
                    pos_list, ptrs_list = zip(*pos_and_ptrs)
                    # stack object pointers along dim=0 into [ptr_seq_len, B, C] shape
                    obj_ptrs = torch.stack(ptrs_list, dim=0) # [1, 1, 256]
                    # a temporal positional embedding based on how far each object pointer is from
                    # the current frame (sine embedding normalized by the max pointer num).
                    if self.add_tpos_enc_to_obj_ptrs: # True
                        t_diff_max = max_obj_ptrs_in_encoder - 1 # 7
                        tpos_dim = C if self.proj_tpos_enc_in_obj_ptrs else self.mem_dim
                        obj_pos = torch.tensor(pos_list, device=device)
                        obj_pos = get_1d_sine_pe(obj_pos / t_diff_max, dim=tpos_dim)
                        obj_pos = self.obj_ptr_tpos_proj(obj_pos)
                        obj_pos = obj_pos.unsqueeze(1).expand(-1, B, self.mem_dim) # 给obj_pointer加上positional coding
                    else:
                        obj_pos = obj_ptrs.new_zeros(len(pos_list), B, self.mem_dim)
                    if self.mem_dim < C:
                        # split a pointer into (C // self.mem_dim) tokens for self.mem_dim < C
                        obj_ptrs = obj_ptrs.reshape(
                            -1, B, C // self.mem_dim, self.mem_dim
                        )
                        obj_ptrs = obj_ptrs.permute(0, 2, 1, 3).flatten(0, 1) #[4, 1, 64]
                        obj_pos = obj_pos.repeat_interleave(C // self.mem_dim, dim=0)
                    to_cat_memory.append(obj_ptrs)
                    exo_memories.append(obj_ptrs)
                    to_cat_memory_pos_embed.append(obj_pos)
                    exo_memories_pos.append(obj_pos)
                    num_obj_ptr_tokens = obj_ptrs.shape[0]
                else:
                    num_obj_ptr_tokens = 0
        else:
            # for initial conditioning frames, encode them without using any previous memory
            if self.directly_add_no_mem_embed:
                # directly add no-mem embedding (instead of using the transformer encoder)
                pix_feat_with_mem = exo_current_vision_feats[-1] + self.no_mem_embed
                pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
                return pix_feat_with_mem

        current_ego_maskmem_features, current_ego_maskmem_pos_enc = self._encode_new_memory(
            current_vision_feats=ego_current_vision_feats,
            feat_sizes=feat_sizes,
            pred_masks_high_res=ego_gt_masks,
            object_score_logits=torch.tensor([[1.0]]).to("cuda"),  # 跟object_score_logits有什么关系？？
            is_mask_from_pts=False,
        )
        current_ego_maskmem_features = current_ego_maskmem_features.flatten(2).permute(2, 0, 1)
        current_ego_maskmem_pos_enc = current_ego_maskmem_pos_enc[-1].flatten(2).permute(2, 0, 1)
        pix_feat_with_other_view = self.memory_attention(
            curr=exo_current_vision_feats, #[900, 1, 256]
            curr_pos=exo_current_vision_pos_embeds,
            memory=current_ego_maskmem_features,
            memory_pos=current_ego_maskmem_pos_enc,
            num_obj_ptr_tokens=0,
        ) # [hw, B, C]

        if ego_memories != [] and exo_memories != []:
            ego_memory = torch.cat(ego_memories, dim=0)
            exo_memory = torch.cat(exo_memories, dim=0)
            ego_memories_pos = torch.cat(ego_memories_pos, dim=0)
            exo_memories_pos = torch.cat(exo_memories_pos, dim=0)
            fused_memory = torch.cat((ego_memory, exo_memory), dim=0)
            fused_memory_pos_embed = torch.cat((ego_memories_pos, exo_memories_pos), dim=0)
            pix_feat_with_mem = self.memory_attention(
                curr=exo_current_vision_feats,  # [900, 1, 256]
                curr_pos=exo_current_vision_pos_embeds,
                memory=fused_memory,
                memory_pos=fused_memory_pos_embed,
                num_obj_ptr_tokens=num_obj_ptr_tokens,
            ) # [hw, B, C]

            pix_feat_with_mem = self.MV_MoE(pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W),
                                           pix_feat_with_other_view.permute(1, 2, 0).view(B, C, H, W))

        else:
            pix_feat_with_mem = pix_feat_with_other_view.permute(1, 2, 0).view(B, C, H, W)

        return pix_feat_with_mem, None, None, None, None, None


    def _prepare_memory_conditioned_features_compress_wo_prompt(
        self,
        frame_idx,
        is_init_cond_frame,
        ego_current_vision_feats,
        exo_current_vision_feats,
        ego_current_vision_pos_embeds,
        exo_current_vision_pos_embeds,
        feat_sizes,
        ego_output_dict,
        exo_output_dict,
        num_frames,
        track_in_reverse=False,  # tracking in reverse time order (for demo usage)
        ego_gt_masks=None,
        ego_memories=None,
        exo_memories=None,
        ego_memories_pos=None,
        exo_memories_pos=None,
    ):
        """Fuse the current frame's visual feature map with previous memory."""
        B = exo_current_vision_feats[-1].size(1)  # batch size on this frame
        C = self.hidden_dim # 256
        H, W = feat_sizes[-1]  # H/16, W/16 # top-level (lowest-resolution) feature size
        device = exo_current_vision_feats[-1].device
        # The case of `self.num_maskmem == 0` below is primarily used for reproducing SAM on images.
        # In this case, we skip the fusion with any memory.
        if self.num_maskmem == 0:  # 7 Disable memory and skip fusion
            exo_pix_feat = exo_current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
            return exo_pix_feat, None, None, None, None, None
        num_obj_ptr_tokens = 0
        tpos_sign_mul = -1 if track_in_reverse else 1
        if len(exo_output_dict["non_cond_frame_outputs"]) <self.num_maskmem:
            # Step 1: condition the visual features of the current frame on previous memories
            if not is_init_cond_frame:
                # Retrieve the memories encoded with the maskmem backbone
                to_cat_memory, to_cat_memory_pos_embed = [], []
                ego_memories, exo_memories = [], []
                ego_memories_pos, exo_memories_pos = [], []
                cond_outputs = exo_output_dict["cond_frame_outputs"]
                selected_cond_outputs, unselected_cond_outputs = select_closest_cond_frames(
                    frame_idx, cond_outputs, self.max_cond_frames_in_attn
                ) # {} {}

                t_pos_and_prevs = [(0, out) for out in selected_cond_outputs.values()]
                # Add last (self.num_maskmem - 1) frames before current frame for non-conditioning memory
                # the earliest one has t_pos=1 and the latest one has t_pos=self.num_maskmem-1
                # We also allow taking the memory frame non-consecutively (with stride>1), in which case
                # we take (self.num_maskmem - 2) frames among every stride-th frames plus the last frame.
                stride = 1 if self.training else self.memory_temporal_stride_for_eval
                for t_pos in range(1, self.num_maskmem):
                    # [1, 2, 3, 4, 5, 6]
                    t_rel = self.num_maskmem - t_pos # [6, 5, 4, 3, 2, 1]
                    # how many frames before current frame
                    if t_rel == 1:
                        # for t_rel == 1, we take the last frame (regardless of r)
                        if not track_in_reverse:
                            # the frame immediately before this frame (i.e. frame_idx - 1)
                            prev_frame_idx = frame_idx - t_rel
                        else:
                            # the frame immediately after this frame (i.e. frame_idx + 1)
                            prev_frame_idx = frame_idx + t_rel
                    else:
                        # for t_rel >= 2, we take the memory frame from every r-th frames
                        if not track_in_reverse:
                            # first find the nearest frame among every r-th frames before this frame
                            # for r=1, this would be (frame_idx - 2)

                            prev_frame_idx = ((frame_idx - 2) // stride) * stride # [-1] [0]
                            # then seek further among every r-th frames
                            prev_frame_idx = prev_frame_idx - (t_rel - 2) * stride # [-5, -4, -3, -2, -1, 0] [-4, -3, -2, -1, 0, 1]
                        else:
                            # first find the nearest frame among every r-th frames after this frame
                            # for r=1, this would be (frame_idx + 2)
                            prev_frame_idx = -(-(frame_idx + 2) // stride) * stride
                            # then seek further among every r-th frames
                            prev_frame_idx = prev_frame_idx + (t_rel - 2) * stride
                    exo_out = exo_output_dict["non_cond_frame_outputs"].get(prev_frame_idx, None)
                    ego_out = ego_output_dict["non_cond_frame_outputs"].get(prev_frame_idx, None)
                    t_pos_and_prevs.append((t_pos, exo_out, ego_out))

                for t_pos, prev_exo, prev_ego in t_pos_and_prevs:
                    if prev_exo is None:
                        continue  # skip padding frames
                    # "maskmem_features" might have been offloaded to CPU in demo use cases,
                    # so we load it back to GPU (it's a no-op if it's already on GPU).
                    exo_feats = prev_exo["maskmem_features"].to(device, non_blocking=True)
                    ego_feats = prev_ego["maskmem_features"].to(device, non_blocking=True)
                    ego_memories.append(ego_feats.flatten(2).permute(2, 0, 1))
                    exo_memories.append(exo_feats.flatten(2).permute(2, 0, 1))
                    # Spatial positional encoding (it might have been offloaded to CPU in eval)
                    exo_maskmem_enc = prev_exo["maskmem_pos_enc"][-1].to(device)
                    B0, C0, H0, W0 = exo_maskmem_enc.shape
                    exo_maskmem_enc = exo_maskmem_enc.flatten(2).permute(2, 0, 1)

                    ego_maskmem_enc = prev_ego["maskmem_pos_enc"][-1].to(device)
                    ego_maskmem_enc = ego_maskmem_enc.flatten(2).permute(2, 0, 1)

                    exo_memories_pos.append(exo_maskmem_enc)
                    ego_memories_pos.append(ego_maskmem_enc)

            else:
                # for initial conditioning frames, encode them without using any previous memory
                if self.directly_add_no_mem_embed:
                    # directly add no-mem embedding (instead of using the transformer encoder)
                    pix_feat_with_mem = exo_current_vision_feats[-1] + self.no_mem_embed
                    pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
                    return pix_feat_with_mem

        else:
            prev_exo = exo_output_dict["non_cond_frame_outputs"].get(frame_idx-1, None)
            prev_ego = ego_output_dict["non_cond_frame_outputs"].get(frame_idx-1, None)
            if prev_exo is not None and prev_ego is not None:
                exo_feats = prev_exo["maskmem_features"].to(device, non_blocking=True)
                ego_feats = prev_ego["maskmem_features"].to(device, non_blocking=True)
                ego_memories.append(ego_feats.flatten(2).permute(2, 0, 1))
                exo_memories.append(exo_feats.flatten(2).permute(2, 0, 1))
                exo_maskmem_enc = prev_exo["maskmem_pos_enc"][-1].to(device)
                exo_maskmem_enc = exo_maskmem_enc.flatten(2).permute(2, 0, 1)

                ego_maskmem_enc = prev_ego["maskmem_pos_enc"][-1].to(device)
                ego_maskmem_enc = ego_maskmem_enc.flatten(2).permute(2, 0, 1)
                exo_memories_pos.append(exo_maskmem_enc)
                ego_memories_pos.append(ego_maskmem_enc)

                ego_memories, ego_memories_pos = memory_bank_compress(torch.stack(ego_memories, dim=0), torch.stack(ego_memories_pos, dim=0))
                exo_memories, exo_memories_pos = memory_bank_compress(torch.stack(exo_memories, dim=0), torch.stack(exo_memories_pos, dim=0))

                ego_memories = list(torch.unbind(ego_memories.permute(1, 2, 0, 3), dim=0))
                exo_memories = list(torch.unbind(exo_memories.permute(1, 2, 0, 3), dim=0))
                ego_memories_pos = list(torch.unbind(ego_memories_pos.permute(1, 2, 0, 3), dim=0))
                exo_memories_pos = list(torch.unbind(exo_memories_pos.permute(1, 2, 0, 3), dim=0))

        exo_memories1 = [mem.clone() for mem in exo_memories]
        exo_memories_pos1 = [mem.clone() for mem in exo_memories_pos]
        for i, mem_pos in enumerate(exo_memories_pos1):
            exo_memories_pos1[i] = mem_pos + self.maskmem_tpos_enc[self.num_maskmem - i - 2]
        ego_memories_pos1 = [mem.clone() for mem in ego_memories_pos]
        for i, mem_pos in enumerate(ego_memories_pos1):
            ego_memories_pos1[i] = mem_pos + self.maskmem_tpos_enc[self.num_maskmem - i - 2]

        if self.use_obj_ptrs_in_encoder:
            max_obj_ptrs_in_encoder = min(num_frames, self.max_obj_ptrs_in_encoder)
            # First add those object pointers from selected conditioning frames
            # (optionally, only include object pointers in the past during evaluation)

            ptr_cond_outputs = {}
            pos_and_ptrs = [
                # Temporal pos encoding contains how far away each pointer is from current frame
                (
                    (
                        (frame_idx - t) * tpos_sign_mul
                        if self.use_signed_tpos_enc_to_obj_ptrs
                        else abs(frame_idx - t)
                    ),
                    out["obj_ptr"],
                )
                for t, out in ptr_cond_outputs.items()
            ]

            # Add up to (max_obj_ptrs_in_encoder - 1) non-conditioning frames before current frame
            for t_diff in range(1, max_obj_ptrs_in_encoder):  # 1<= t_diff <8
                t = frame_idx + t_diff if track_in_reverse else frame_idx - t_diff
                if t < 0 or (num_frames is not None and t >= num_frames):
                    break
                out = exo_output_dict["non_cond_frame_outputs"].get(
                    t, None
                )
                if out is not None:
                    pos_and_ptrs.append((t_diff, out["obj_ptr"]))
            # If we have at least one object pointer, add them to the across attention
            if len(pos_and_ptrs) > 0:
                pos_list, ptrs_list = zip(*pos_and_ptrs)
                # stack object pointers along dim=0 into [ptr_seq_len, B, C] shape
                obj_ptrs = torch.stack(ptrs_list, dim=0)  # [1, 1, 256]
                # a temporal positional embedding based on how far each object pointer is from
                # the current frame (sine embedding normalized by the max pointer num).
                if self.add_tpos_enc_to_obj_ptrs:  # True
                    t_diff_max = max_obj_ptrs_in_encoder - 1  # 7
                    tpos_dim = C if self.proj_tpos_enc_in_obj_ptrs else self.mem_dim
                    obj_pos = torch.tensor(pos_list, device=device)
                    obj_pos = get_1d_sine_pe(obj_pos / t_diff_max, dim=tpos_dim)
                    obj_pos = self.obj_ptr_tpos_proj(obj_pos)
                    obj_pos = obj_pos.unsqueeze(1).expand(-1, B, self.mem_dim)
                else:
                    obj_pos = obj_ptrs.new_zeros(len(pos_list), B, self.mem_dim)
                if self.mem_dim < C:
                    # split a pointer into (C // self.mem_dim) tokens for self.mem_dim < C
                    obj_ptrs = obj_ptrs.reshape(
                        -1, B, C // self.mem_dim, self.mem_dim
                    )
                    obj_ptrs = obj_ptrs.permute(0, 2, 1, 3).flatten(0, 1)  # [4, 1, 64]
                    obj_pos = obj_pos.repeat_interleave(C // self.mem_dim, dim=0)

                exo_memories1.append(obj_ptrs)
                exo_memories_pos1.append(obj_pos)
                num_obj_ptr_tokens = obj_ptrs.shape[0]
            else:
                num_obj_ptr_tokens = 0

        pix_feat_with_mem1 = None



        current_ego_maskmem_features, current_ego_maskmem_pos_enc = self._encode_new_memory(
            current_vision_feats=ego_current_vision_feats,
            feat_sizes=feat_sizes,
            pred_masks_high_res=ego_gt_masks,
            object_score_logits=torch.tensor([[1.0]]).to("cuda"),
            is_mask_from_pts=False,
        )
        current_ego_maskmem_features = current_ego_maskmem_features.flatten(2).permute(2, 0, 1)
        current_ego_maskmem_pos_enc = current_ego_maskmem_pos_enc[-1].flatten(2).permute(2, 0, 1)
        pix_feat_with_other_view = self.memory_attention(
            curr=exo_current_vision_feats, #[900, 1, 256]
            curr_pos=exo_current_vision_pos_embeds,
            memory=current_ego_maskmem_features,
            memory_pos=current_ego_maskmem_pos_enc,
            num_obj_ptr_tokens=0,
        ) # [hw, B, C]

        if ego_memories != [] and exo_memories1 != []:
            ego_memory = torch.cat(ego_memories, dim=0)
            exo_memory = torch.cat(exo_memories1, dim=0)
            ego_memories_pos_final = torch.cat(ego_memories_pos1, dim=0)
            exo_memories_pos_final = torch.cat(exo_memories_pos1, dim=0)
            fused_memory = torch.cat((ego_memory, exo_memory), dim=0)
            fused_memory_pos_embed = torch.cat((ego_memories_pos_final, exo_memories_pos_final), dim=0)
            pix_feat_with_mem = self.memory_attention(
                curr=exo_current_vision_feats,
                curr_pos=exo_current_vision_pos_embeds,
                memory=fused_memory,
                memory_pos=fused_memory_pos_embed,
                num_obj_ptr_tokens=num_obj_ptr_tokens,
            ) # [hw, B, C]


            pix_feat_with_mem = self.MV_MoE(pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W), pix_feat_with_other_view.permute(1, 2, 0).view(B, C, H, W))

        else:
            pix_feat_with_mem = pix_feat_with_other_view.permute(1, 2, 0).view(B, C, H, W)

        return pix_feat_with_mem, pix_feat_with_mem1, ego_memories, exo_memories, ego_memories_pos, exo_memories_pos

    def _prepare_memory_conditioned_features_wo_prompt_cluster_selection(
        self,
        frame_idx,
        is_init_cond_frame,
        ego_current_vision_feats,
        exo_current_vision_feats,
        ego_current_vision_pos_embeds,
        exo_current_vision_pos_embeds,
        feat_sizes,
        ego_output_dict,
        exo_output_dict,
        num_frames,
        track_in_reverse=False,  # tracking in reverse time order (for demo usage)
        ego_gt_masks=None,
        ego_memories=None,
        exo_memories=None,
        ego_memories_pos=None,
        exo_memories_pos=None,
    ):
        """Fuse the current frame's visual feature map with previous memory."""
        B = exo_current_vision_feats[-1].size(1)  # batch size on this frame
        C = self.hidden_dim # 256
        H, W = feat_sizes[-1]  # H/16, W/16 # top-level (lowest-resolution) feature size
        device = exo_current_vision_feats[-1].device
        # The case of `self.num_maskmem == 0` below is primarily used for reproducing SAM on images.
        # In this case, we skip the fusion with any memory.
        if self.num_maskmem == 0:  # 7 Disable memory and skip fusion
            exo_pix_feat = exo_current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
            return exo_pix_feat, None, None, None, None, None
        num_obj_ptr_tokens = 0
        tpos_sign_mul = -1 if track_in_reverse else 1
        if len(exo_output_dict["non_cond_frame_outputs"]) <self.num_maskmem:
            # Step 1: condition the visual features of the current frame on previous memories
            if not is_init_cond_frame:
                # Retrieve the memories encoded with the maskmem backbone
                to_cat_memory, to_cat_memory_pos_embed = [], []
                ego_memories, exo_memories = [], []
                ego_memories_pos, exo_memories_pos = [], []
                cond_outputs = exo_output_dict["cond_frame_outputs"]
                selected_cond_outputs, unselected_cond_outputs = select_closest_cond_frames(
                    frame_idx, cond_outputs, self.max_cond_frames_in_attn
                ) # {} {}
                t_pos_and_prevs = [(0, out) for out in selected_cond_outputs.values()]
                # Add last (self.num_maskmem - 1) frames before current frame for non-conditioning memory
                # the earliest one has t_pos=1 and the latest one has t_pos=self.num_maskmem-1
                # We also allow taking the memory frame non-consecutively (with stride>1), in which case
                # we take (self.num_maskmem - 2) frames among every stride-th frames plus the last frame.
                stride = 1 if self.training else self.memory_temporal_stride_for_eval
                for t_pos in range(1, self.num_maskmem):
                    # [1, 2, 3, 4, 5, 6]
                    t_rel = self.num_maskmem - t_pos # [6, 5, 4, 3, 2, 1]
                    # how many frames before current frame
                    if t_rel == 1:
                        # for t_rel == 1, we take the last frame (regardless of r)
                        if not track_in_reverse:
                            # the frame immediately before this frame (i.e. frame_idx - 1)
                            prev_frame_idx = frame_idx - t_rel
                        else:
                            # the frame immediately after this frame (i.e. frame_idx + 1)
                            prev_frame_idx = frame_idx + t_rel
                    else:
                        # for t_rel >= 2, we take the memory frame from every r-th frames
                        if not track_in_reverse:
                            prev_frame_idx = ((frame_idx - 2) // stride) * stride # [-1] [0]
                            # then seek further among every r-th frames
                            prev_frame_idx = prev_frame_idx - (t_rel - 2) * stride # [-5, -4, -3, -2, -1, 0] [-4, -3, -2, -1, 0, 1]
                        else:
                            # first find the nearest frame among every r-th frames after this frame
                            # for r=1, this would be (frame_idx + 2)
                            prev_frame_idx = -(-(frame_idx + 2) // stride) * stride
                            # then seek further among every r-th frames
                            prev_frame_idx = prev_frame_idx + (t_rel - 2) * stride
                    exo_out = exo_output_dict["non_cond_frame_outputs"].get(prev_frame_idx, None)
                    ego_out = ego_output_dict["non_cond_frame_outputs"].get(prev_frame_idx, None)
                    # if exo_out is None:
                    #     # If an unselected conditioning frame is among the last (self.num_maskmem - 1)
                    #     # frames, we still attend to it as if it's a non-conditioning frame.
                    #     out = unselected_cond_outputs.get(prev_frame_idx, None)
                    t_pos_and_prevs.append((t_pos, exo_out, ego_out))

                for t_pos, prev_exo, prev_ego in t_pos_and_prevs:
                    if prev_exo is None:
                        continue  # skip padding frames
                    # "maskmem_features" might have been offloaded to CPU in demo use cases,
                    # so we load it back to GPU (it's a no-op if it's already on GPU).
                    exo_feats = prev_exo["maskmem_features"].to(device, non_blocking=True)
                    ego_feats = prev_ego["maskmem_features"].to(device, non_blocking=True)
                    ego_memories.append(ego_feats.flatten(2).permute(2, 0, 1))
                    exo_memories.append(exo_feats.flatten(2).permute(2, 0, 1))
                    # Spatial positional encoding (it might have been offloaded to CPU in eval)
                    exo_maskmem_enc = prev_exo["maskmem_pos_enc"][-1].to(device)
                    B0, C0, H0, W0 = exo_maskmem_enc.shape
                    exo_maskmem_enc = exo_maskmem_enc.flatten(2).permute(2, 0, 1)

                    ego_maskmem_enc = prev_ego["maskmem_pos_enc"][-1].to(device)
                    ego_maskmem_enc = ego_maskmem_enc.flatten(2).permute(2, 0, 1)

                    exo_memories_pos.append(exo_maskmem_enc)
                    ego_memories_pos.append(ego_maskmem_enc)

            else:
                # for initial conditioning frames, encode them without using any previous memory
                if self.directly_add_no_mem_embed:
                    # directly add no-mem embedding (instead of using the transformer encoder)
                    pix_feat_with_mem = exo_current_vision_feats[-1] + self.no_mem_embed
                    pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
                    return pix_feat_with_mem

        else:
            ego_non_cond_frame_outputs = ego_output_dict['non_cond_frame_outputs']

            ego_features_list = [v['maskmem_features'].unsqueeze(1) for k, v in ego_non_cond_frame_outputs.items()]
            ego_pos_list = [v["maskmem_pos_enc"][-1].unsqueeze(1) for k, v in ego_non_cond_frame_outputs.items()]
            ego_all_features = torch.cat(ego_features_list, dim=1).to(device, non_blocking=True)
            ego_all_pos = torch.cat(ego_pos_list, dim=1).to(device, non_blocking=True)

            exo_non_cond_frame_outputs = exo_output_dict['non_cond_frame_outputs']

            exo_features_list = [v['maskmem_features'].unsqueeze(1) for k, v in exo_non_cond_frame_outputs.items()]
            exo_pos_list = [v["maskmem_pos_enc"][-1].unsqueeze(1) for k, v in exo_non_cond_frame_outputs.items()]

            exo_all_features = torch.cat(exo_features_list, dim=1).to(device, non_blocking=True)
            exo_all_pos = torch.cat(exo_pos_list, dim=1).to(device, non_blocking=True)

            exo_all_features_pool = exo_all_features.mean(dim=[3, 4])
            batch_keyframes_index = kmeans_cluster_frame(exo_all_features_pool, 6)

            exo_selected_features = torch.stack([
                exo_all_features[b, frame_indices]  # shape: [N, C]
                for b, frame_indices in enumerate(batch_keyframes_index)
            ]).flatten(3).permute(3, 0, 1, 2)
            ego_selected_features = torch.stack([
                ego_all_features[b, frame_indices]  # shape: [N, C]
                for b, frame_indices in enumerate(batch_keyframes_index)
            ]).flatten(3).permute(3, 0, 1, 2)
            exo_selected_pos = torch.stack([
                exo_all_pos[b, frame_indices]  # shape: [N, C]
                for b, frame_indices in enumerate(batch_keyframes_index)
            ]).flatten(3).permute(3, 0, 1, 2)
            ego_selected_pos = torch.stack([
                ego_all_pos[b, frame_indices]  # shape: [N, C]
                for b, frame_indices in enumerate(batch_keyframes_index)
            ]).flatten(3).permute(3, 0, 1, 2)

            ego_memories = list(torch.unbind(ego_selected_features, dim=2))
            exo_memories = list(torch.unbind(exo_selected_features, dim=2))
            ego_memories_pos = list(torch.unbind(ego_selected_pos, dim=2))
            exo_memories_pos = list(torch.unbind(exo_selected_pos, dim=2))

        exo_memories1 = [mem.clone() for mem in exo_memories]
        exo_memories_pos1 = [mem.clone() for mem in exo_memories_pos]
        for i, mem_pos in enumerate(exo_memories_pos1):
            exo_memories_pos1[i] = mem_pos + self.maskmem_tpos_enc[self.num_maskmem - i - 2]
        ego_memories_pos1 = [mem.clone() for mem in ego_memories_pos]
        for i, mem_pos in enumerate(ego_memories_pos1):
            ego_memories_pos1[i] = mem_pos + self.maskmem_tpos_enc[self.num_maskmem - i - 2]
        if self.use_obj_ptrs_in_encoder:
            max_obj_ptrs_in_encoder = min(num_frames, self.max_obj_ptrs_in_encoder)
            ptr_cond_outputs = {}
            pos_and_ptrs = [
                # Temporal pos encoding contains how far away each pointer is from current frame
                (
                    (
                        (frame_idx - t) * tpos_sign_mul
                        if self.use_signed_tpos_enc_to_obj_ptrs
                        else abs(frame_idx - t)
                    ),
                    out["obj_ptr"],
                )
                for t, out in ptr_cond_outputs.items()
            ]
            # Add up to (max_obj_ptrs_in_encoder - 1) non-conditioning frames before current frame
            for t_diff in range(1, max_obj_ptrs_in_encoder):  # 1<= t_diff <8
                t = frame_idx + t_diff if track_in_reverse else frame_idx - t_diff
                if t < 0 or (num_frames is not None and t >= num_frames):
                    break
                out = exo_output_dict["non_cond_frame_outputs"].get(
                    t, None
                )
                if out is not None:
                    pos_and_ptrs.append((t_diff, out["obj_ptr"]))
            # If we have at least one object pointer, add them to the across attention
            if len(pos_and_ptrs) > 0:
                pos_list, ptrs_list = zip(*pos_and_ptrs)
                # stack object pointers along dim=0 into [ptr_seq_len, B, C] shape
                obj_ptrs = torch.stack(ptrs_list, dim=0)  # [1, 1, 256]
                # a temporal positional embedding based on how far each object pointer is from
                # the current frame (sine embedding normalized by the max pointer num).
                if self.add_tpos_enc_to_obj_ptrs:  # True
                    t_diff_max = max_obj_ptrs_in_encoder - 1  # 7
                    tpos_dim = C if self.proj_tpos_enc_in_obj_ptrs else self.mem_dim
                    obj_pos = torch.tensor(pos_list, device=device)
                    obj_pos = get_1d_sine_pe(obj_pos / t_diff_max, dim=tpos_dim)
                    obj_pos = self.obj_ptr_tpos_proj(obj_pos)
                    obj_pos = obj_pos.unsqueeze(1).expand(-1, B, self.mem_dim)
                else:
                    obj_pos = obj_ptrs.new_zeros(len(pos_list), B, self.mem_dim)
                if self.mem_dim < C:
                    # split a pointer into (C // self.mem_dim) tokens for self.mem_dim < C
                    obj_ptrs = obj_ptrs.reshape(
                        -1, B, C // self.mem_dim, self.mem_dim
                    )
                    obj_ptrs = obj_ptrs.permute(0, 2, 1, 3).flatten(0, 1)  # [4, 1, 64]
                    obj_pos = obj_pos.repeat_interleave(C // self.mem_dim, dim=0)

                exo_memories1.append(obj_ptrs)
                exo_memories_pos1.append(obj_pos)
                num_obj_ptr_tokens = obj_ptrs.shape[0]
            else:
                num_obj_ptr_tokens = 0

        pix_feat_with_mem1 = None



        current_ego_maskmem_features, current_ego_maskmem_pos_enc = self._encode_new_memory(
            current_vision_feats=ego_current_vision_feats,
            feat_sizes=feat_sizes,
            pred_masks_high_res=ego_gt_masks,
            object_score_logits=torch.tensor([[1.0]]).to("cuda"),  # 跟object_score_logits有什么关系？？
            is_mask_from_pts=False,
        )
        current_ego_maskmem_features = current_ego_maskmem_features.flatten(2).permute(2, 0, 1)
        current_ego_maskmem_pos_enc = current_ego_maskmem_pos_enc[-1].flatten(2).permute(2, 0, 1)
        pix_feat_with_other_view = self.memory_attention(
            curr=exo_current_vision_feats, #[900, 1, 256]
            curr_pos=exo_current_vision_pos_embeds,
            memory=current_ego_maskmem_features,
            memory_pos=current_ego_maskmem_pos_enc,
            num_obj_ptr_tokens=0,
        ) # [hw, B, C]

        if ego_memories != [] and exo_memories1 != []:
            ego_memory = torch.cat(ego_memories, dim=0)
            exo_memory = torch.cat(exo_memories1, dim=0)
            ego_memories_pos_final = torch.cat(ego_memories_pos1, dim=0)
            exo_memories_pos_final = torch.cat(exo_memories_pos1, dim=0)
            fused_memory = torch.cat((ego_memory, exo_memory), dim=0)
            fused_memory_pos_embed = torch.cat((ego_memories_pos_final, exo_memories_pos_final), dim=0)
            pix_feat_with_mem = self.memory_attention(
                curr=exo_current_vision_feats,  # [900, 1, 256]
                curr_pos=exo_current_vision_pos_embeds,
                memory=fused_memory,
                memory_pos=fused_memory_pos_embed,
                num_obj_ptr_tokens=num_obj_ptr_tokens,
            ) # [hw, B, C]



            # pix_feat_with_mem = self.fuser(pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W), pix_feat_with_other_view.permute(1, 2, 0).view(B, C, H, W))
            pix_feat_with_mem = self.MV_MoE(pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W),
                                           pix_feat_with_other_view.permute(1, 2, 0).view(B, C, H, W))

        else:
            pix_feat_with_mem = pix_feat_with_other_view.permute(1, 2, 0).view(B, C, H, W)

        return pix_feat_with_mem, pix_feat_with_mem1, ego_memories, exo_memories, ego_memories_pos, exo_memories_pos

    def _prepare_memory_conditioned_features_wo_prompt_IoU_selection(
            self,
            frame_idx,
            is_init_cond_frame,
            ego_current_vision_feats,
            exo_current_vision_feats,
            ego_current_vision_pos_embeds,
            exo_current_vision_pos_embeds,
            feat_sizes,
            ego_output_dict,
            exo_output_dict,
            num_frames,
            track_in_reverse=False,  # tracking in reverse time order (for demo usage)
            ego_gt_masks=None,
            iou_thre=0.3,
    ):
        """Fuse the current frame's visual feature map with previous memory."""
        B = exo_current_vision_feats[-1].size(1)  # batch size on this frame
        C = self.hidden_dim  # 256
        H, W = feat_sizes[-1]  # H/16, W/16 # top-level (lowest-resolution) feature size
        device = exo_current_vision_feats[-1].device
        num_obj_ptr_tokens = 0
        tpos_sign_mul = -1 if track_in_reverse else 1
        # Step 1: condition the visual features of the current frame on previous memories
        if not is_init_cond_frame:
            # Retrieve the memories encoded with the maskmem backbone
            to_cat_memory, to_cat_memory_pos_embed = [], []
            # Add conditioning frames's output first (all cond frames have t_pos=0 for
            # when getting temporal positional embedding below)
            # assert len(exo_output_dict["cond_frame_outputs"]) > 0
            # Select a maximum number of temporally closest cond frames for cross attention
            cond_outputs = exo_output_dict["cond_frame_outputs"]
            selected_cond_outputs, unselected_cond_outputs = select_closest_cond_frames(
                frame_idx, cond_outputs, self.max_cond_frames_in_attn
            )  # {} {}
            t_pos_and_prevs = [(0, out) for out in selected_cond_outputs.values()]
            max_obj_ptrs_in_encoder = min(num_frames, self.max_obj_ptrs_in_encoder)
            # Add last (self.num_maskmem - 1) frames before current frame for non-conditioning memory
            # the earliest one has t_pos=1 and the latest one has t_pos=self.num_maskmem-1
            # We also allow taking the memory frame non-consecutively (with stride>1), in which case
            # we take (self.num_maskmem - 2) frames among every stride-th frames plus the last frame.
            stride = 1 if self.training else self.memory_temporal_stride_for_eval
            valid_indices = []
            for i in range(frame_idx - 1, 0, -1):
                if i not in exo_output_dict["non_cond_frame_outputs"]:
                    continue

                if "iou" in exo_output_dict["non_cond_frame_outputs"][i]:
                    iou = exo_output_dict["non_cond_frame_outputs"][i]["iou"]
                else:
                    continue
                if iou.item() > iou_thre:
                    valid_indices.insert(0, i)
                if len(valid_indices) >= max_obj_ptrs_in_encoder - 1:
                    break
            if frame_idx - 1 not in valid_indices:  ##pick last frame
                valid_indices.append(frame_idx - 1)

            if frame_idx - 2 not in valid_indices:  ##pick last frame
                valid_indices.append(frame_idx - 2)
        prev_idxs = []
        for t_pos in range(1, self.num_maskmem):
            idx = t_pos - self.num_maskmem
            if idx < -len(valid_indices):
                continue
            exo_out = exo_output_dict["non_cond_frame_outputs"].get(valid_indices[idx], None)
            ego_out = ego_output_dict["non_cond_frame_outputs"].get(valid_indices[idx], None)
            # if out is None:
            #     out = unselected_cond_outputs.get(valid_indices[idx], None)
            t_pos_and_prevs.append((t_pos, exo_out, ego_out))
            prev_idxs.append(valid_indices[idx])

        for (t_pos, exo_prev, ego_prev), prev_idx in zip(t_pos_and_prevs, prev_idxs):
            if exo_prev is None:
                continue  # skip padding frames
            # "maskmem_features" might have been offloaded to CPU in demo use cases,
            # so we load it back to GPU (it's a no-op if it's already on GPU).
            exo_feats = exo_prev["maskmem_features"].to(device, non_blocking=True)
            ego_feats = ego_prev["maskmem_features"].to(device, non_blocking=True)
            to_cat_memory.append(exo_feats.flatten(2).permute(2, 0, 1))
            to_cat_memory.append(ego_feats.flatten(2).permute(2, 0, 1))
            # Spatial positional encoding (it might have been offloaded to CPU in eval)
            exo_maskmem_enc = exo_prev["maskmem_pos_enc"][-1].to(device)
            exo_maskmem_enc = exo_maskmem_enc.flatten(2).permute(2, 0, 1)
            # Temporal positional encoding
            exo_maskmem_enc = (
                    exo_maskmem_enc + self.maskmem_tpos_enc[self.num_maskmem - t_pos - 1]
            )
            to_cat_memory_pos_embed.append(exo_maskmem_enc)

            ego_maskmem_enc = ego_prev["maskmem_pos_enc"][-1].to(device)
            ego_maskmem_enc = ego_maskmem_enc.flatten(2).permute(2, 0, 1)
            # Temporal positional encoding
            ego_maskmem_enc = (
                    ego_maskmem_enc + self.maskmem_tpos_enc[self.num_maskmem - t_pos - 1]
            )
            to_cat_memory_pos_embed.append(ego_maskmem_enc)
        # Construct the list of past object pointers
        if self.use_obj_ptrs_in_encoder:
            max_obj_ptrs_in_encoder = min(num_frames, self.max_obj_ptrs_in_encoder)
            object_ptr_score = []
            pos_and_ptrs = []
            for t_diff in range(1, max_obj_ptrs_in_encoder):
                if -t_diff < -len(valid_indices):
                    break
                out = exo_output_dict["non_cond_frame_outputs"].get(
                    valid_indices[-t_diff], unselected_cond_outputs.get(valid_indices[-t_diff], None))
                if out is not None:
                    # mem_idx = mem_pick_index[valid_indices[-t_diff]]
                    object_ptr_score.append(out['object_score_logits'].view(-1))
                    # object_ptr_score.append(out['object_score_logits'].view(-1))
                    pos_and_ptrs.append((t_diff, out["obj_ptr"]))
            # object_ptr_score.append(output_dict["non_cond_frame_outputs"][valid_indices[-t_diff]]['object_score'].item())
            # If we have at least one object pointer, add them to the across attention
            if len(pos_and_ptrs) > 0:
                pos_list, ptrs_list = zip(*pos_and_ptrs)
                # stack object pointers along dim=0 into [ptr_seq_len, B, C] shape
                obj_ptrs = torch.stack(ptrs_list, dim=0)
                # a temporal positional embedding based on how far each object pointer is from
                # the current frame (sine embedding normalized by the max pointer num).
                if self.add_tpos_enc_to_obj_ptrs:
                    t_diff_max = max_obj_ptrs_in_encoder - 1
                    tpos_dim = C if self.proj_tpos_enc_in_obj_ptrs else self.mem_dim
                    obj_pos = torch.tensor(pos_list, device=device)
                    obj_pos = get_1d_sine_pe(obj_pos / t_diff_max, dim=tpos_dim)
                    obj_pos = self.obj_ptr_tpos_proj(obj_pos)
                    obj_pos = obj_pos.unsqueeze(1).expand(-1, B, self.mem_dim)
                else:
                    obj_pos = obj_ptrs.new_zeros(len(pos_list), B, self.mem_dim)
                if self.mem_dim < C:
                    # split a pointer into (C // self.mem_dim) tokens for self.mem_dim < C
                    obj_ptrs = obj_ptrs.reshape(
                        -1, B, C // self.mem_dim, self.mem_dim
                    )
                    obj_ptrs = obj_ptrs.permute(0, 2, 1, 3).flatten(0, 1)
                    obj_pos = obj_pos.repeat_interleave(C // self.mem_dim, dim=0)
                to_cat_memory.append(obj_ptrs)
                to_cat_memory_pos_embed.append(obj_pos)
                num_obj_ptr_tokens = obj_ptrs.shape[0]
            else:
                num_obj_ptr_tokens = 0

        current_ego_maskmem_features, current_ego_maskmem_pos_enc = self._encode_new_memory(
            current_vision_feats=ego_current_vision_feats,
            feat_sizes=feat_sizes,
            pred_masks_high_res=ego_gt_masks,
            object_score_logits=torch.tensor([[1.0]]).to("cuda"),
            is_mask_from_pts=False,
        )
        current_ego_maskmem_features = current_ego_maskmem_features.flatten(2).permute(2, 0, 1)
        current_ego_maskmem_pos_enc = current_ego_maskmem_pos_enc[-1].flatten(2).permute(2, 0, 1)
        pix_feat_with_other_view = self.memory_attention(
            curr=exo_current_vision_feats, #[900, 1, 256]
            curr_pos=exo_current_vision_pos_embeds,
            memory=current_ego_maskmem_features,
            memory_pos=current_ego_maskmem_pos_enc,
            num_obj_ptr_tokens=0,
        ) # [hw, B, C]

        if to_cat_memory != []:
            memory = torch.cat(to_cat_memory, dim=0)
            memory_pos_embed = torch.cat(to_cat_memory_pos_embed, dim=0)
            pix_feat_with_mem = self.memory_attention(
                curr=exo_current_vision_feats,  # [900, 1, 256]
                curr_pos=exo_current_vision_pos_embeds,
                memory=memory,
                memory_pos=memory_pos_embed,
                num_obj_ptr_tokens=num_obj_ptr_tokens,
            ) # [hw, B, C]

            pix_feat_with_mem = self.MV_MoE(pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W),
                                           pix_feat_with_other_view.permute(1, 2, 0).view(B, C, H, W))

        else:
            pix_feat_with_mem = pix_feat_with_other_view.permute(1, 2, 0).view(B, C, H, W)

        return pix_feat_with_mem, None, None, None, None, None

    def _encode_new_memory(
        self,
        current_vision_feats,
        feat_sizes,
        pred_masks_high_res,
        object_score_logits,
        is_mask_from_pts,
    ):
        """Encode the current image and its prediction into a memory feature."""
        B = current_vision_feats[-1].size(1)  # batch size on this frame
        C = self.hidden_dim
        H, W = feat_sizes[-1]  # top-level (lowest-resolution) feature size
        # top-level feature, (HW)BC => BCHW
        pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
        if self.non_overlap_masks_for_mem_enc and not self.training:
            # optionally, apply non-overlapping constraints to the masks (it's applied
            # in the batch dimension and should only be used during eval, where all
            # the objects come from the same video under batch size 1).
            pred_masks_high_res = self._apply_non_overlapping_constraints(
                pred_masks_high_res
            )
        # scale the raw mask logits with a temperature before applying sigmoid
        binarize = self.binarize_mask_from_pts_for_mem_enc and is_mask_from_pts
        if binarize and not self.training:
            mask_for_mem = (pred_masks_high_res > 0).float()
        else:
            # apply sigmoid on the raw mask logits to turn them into range (0, 1)
            mask_for_mem = torch.sigmoid(pred_masks_high_res)
        # apply scale and bias terms to the sigmoid probabilities
        if self.sigmoid_scale_for_mem_enc != 1.0:
            mask_for_mem = mask_for_mem * self.sigmoid_scale_for_mem_enc
        if self.sigmoid_bias_for_mem_enc != 0.0:
            mask_for_mem = mask_for_mem + self.sigmoid_bias_for_mem_enc
        maskmem_out = self.memory_encoder(
            pix_feat, mask_for_mem, skip_mask_sigmoid=True  # sigmoid already applied
        )
        maskmem_features = maskmem_out["vision_features"]
        maskmem_pos_enc = maskmem_out["vision_pos_enc"]
        # add a no-object embedding to the spatial memory to indicate that the frame
        # is predicted to be occluded (i.e. no object is appearing in the frame)
        if self.no_obj_embed_spatial is not None:
            is_obj_appearing = (object_score_logits > 0).float()
            maskmem_features += (
                1 - is_obj_appearing[..., None, None]
            ) * self.no_obj_embed_spatial[..., None, None].expand(
                *maskmem_features.shape
            )

        return maskmem_features, maskmem_pos_enc
    def _track_step(
        self,
        frame_idx,
        is_init_cond_frame,
        ego_current_vision_feats,
        exo_current_vision_feats,
        ego_current_vision_pos_embeds,
        exo_current_vision_pos_embeds,
        feat_sizes,
        ego_point_inputs, # None
        exo_point_inputs, # None
        ego_mask_inputs,  # None
        exo_mask_inputs,  # None
        ego_output_dict, # {'cond_frame_outputs':{}, 'non_cond_frame_outputs':{}}
        exo_output_dict, # {'cond_frame_outputs':{}, 'non_cond_frame_outputs':{}}
        num_frames,
        track_in_reverse,
        prev_sam_mask_logits, # None
        ego_gt_masks,
        exo_gt_masks=None,
        ego_memories=None,
        exo_memories=None,
        ego_memories_pos=None,
        exo_memories_pos=None,
    ):
        exo_current_out = {"point_inputs": exo_point_inputs, "mask_inputs": exo_mask_inputs} # {'mask_inputs': None, 'point_inputs': None}
        ego_current_out = {"point_inputs": ego_point_inputs, "mask_inputs": ego_mask_inputs} # {'mask_inputs': None, 'point_inputs': None}
        if len(ego_current_vision_feats) > 1:
            ego_high_res_features = [
                x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                for x, s in zip(ego_current_vision_feats[:-1], feat_sizes[:-1])
            ]
        else:
            ego_high_res_features = None
        if len(exo_current_vision_feats) > 1:
            exo_high_res_features = [
                x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                for x, s in zip(exo_current_vision_feats[:-1], feat_sizes[:-1])
            ]
        else:
            exo_high_res_features = None

        if exo_mask_inputs is not None and self.use_mask_input_as_output_without_sam:
            # When use_mask_input_as_output_without_sam=True, we directly output the mask input
            # (see it as a GT mask) without using a SAM prompt encoder + mask decoder.
            pix_feat = exo_current_vision_feats[-1].permute(1, 2, 0) # 【1， 256， 4096】
            pix_feat = pix_feat.view(-1, self.hidden_dim, *feat_sizes[-1]) # 【1， 256，64， 64】
            sam_outputs = self._use_mask_as_output(
                pix_feat, exo_high_res_features, exo_mask_inputs
            )
        else:
            if self.training is True:
                # fused the visual feature with previous memory features in the memory bank
                exo_pix_feat, historical_feature, ego_memories, exo_memories, ego_memories_pos, exo_memories_pos  = self._prepare_memory_conditioned_features_wo_prompt(
                    frame_idx=frame_idx,
                    is_init_cond_frame=is_init_cond_frame,
                    ego_current_vision_feats = ego_current_vision_feats[-1:],
                    exo_current_vision_feats=exo_current_vision_feats[-1:],
                    ego_current_vision_pos_embeds=ego_current_vision_pos_embeds[-1:],
                    exo_current_vision_pos_embeds=exo_current_vision_pos_embeds[-1:],
                    feat_sizes=feat_sizes[-1:],
                    ego_output_dict=ego_output_dict,
                    exo_output_dict=exo_output_dict,
                    num_frames=num_frames,
                    track_in_reverse=track_in_reverse, # False
                    ego_gt_masks = ego_gt_masks,
                )
            else:
                exo_pix_feat, historical_feature, ego_memories, exo_memories, ego_memories_pos, exo_memories_pos = self._prepare_memory_conditioned_features_compress_wo_prompt(
                    frame_idx=frame_idx,
                    is_init_cond_frame=is_init_cond_frame,
                    ego_current_vision_feats = ego_current_vision_feats[-1:],
                    exo_current_vision_feats=exo_current_vision_feats[-1:],
                    ego_current_vision_pos_embeds=ego_current_vision_pos_embeds[-1:],
                    exo_current_vision_pos_embeds=exo_current_vision_pos_embeds[-1:],
                    feat_sizes=feat_sizes[-1:],
                    ego_output_dict=ego_output_dict,
                    exo_output_dict=exo_output_dict,
                    num_frames=num_frames,
                    track_in_reverse=track_in_reverse,
                    ego_gt_masks=ego_gt_masks,
                    ego_memories=ego_memories,
                    exo_memories=exo_memories,
                    ego_memories_pos=ego_memories_pos,
                    exo_memories_pos=exo_memories_pos# False
                )

            B, C, H, W = exo_pix_feat.shape # [1, 256, 24, 24]
            # print("exo_pix_feat.shape:", exo_pix_feat.shape)
            ego_pix_feat = ego_current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
            # apply SAM-style segmentation head
            # here we might feed previously predicted low-res SAM mask logits into the SAM mask decoder,
            # e.g. in demo where such logits come from earlier interaction instead of correction sampling
            # (in this case, any `mask_inputs` shouldn't reach here as they are sent to _use_mask_as_output instead)
            if prev_sam_mask_logits is not None:
                assert exo_point_inputs is not None and exo_mask_inputs is None
                exo_mask_inputs = prev_sam_mask_logits
            multimask_output = False
            sam_outputs = self._forward_sam_heads(
                ego_backbone_features = ego_pix_feat,
                exo_backbone_features = exo_pix_feat,
                exo_point_inputs = exo_point_inputs,
                ego_mask_inputs=ego_mask_inputs,
                exo_mask_inputs=exo_mask_inputs,
                ego_high_res_features=ego_high_res_features,
                exo_high_res_features=exo_high_res_features,
                multimask_output=multimask_output,
                ego_gt_masks=ego_gt_masks,
                exo_gt_masks= exo_gt_masks
            )
        if self.training is False:
            return exo_current_out, ego_current_out, sam_outputs, exo_high_res_features, exo_pix_feat, historical_feature, ego_memories, exo_memories, ego_memories_pos, exo_memories_pos
        return exo_current_out, ego_current_out, sam_outputs, exo_high_res_features, exo_pix_feat, ego_memories, exo_memories, ego_memories_pos, exo_memories_pos

    def _encode_memory_in_output(
        self,
        exo_current_vision_feats,
        feat_sizes,
        exo_point_inputs,
        run_mem_encoder,
        high_res_masks,
        object_score_logits,
        exo_current_out,
        ego_current_out,
        ego_gt_masks,
        ego_current_vision_feats
    ):
        if run_mem_encoder and self.num_maskmem > 0:
            high_res_masks_for_mem_enc = high_res_masks
            # print("exo_current_vision_feats:", exo_current_vision_feats.shape)
            exo_maskmem_features, exo_maskmem_pos_enc = self._encode_new_memory(
                current_vision_feats=exo_current_vision_feats,
                feat_sizes=feat_sizes,
                pred_masks_high_res=high_res_masks_for_mem_enc,
                object_score_logits=object_score_logits,
                is_mask_from_pts=(exo_point_inputs is not None),
            )

            # print("object_score_logits:", object_score_logits)
            exo_current_out["maskmem_features"] = exo_maskmem_features
            exo_current_out["maskmem_pos_enc"] = exo_maskmem_pos_enc
            ego_maskmem_features, ego_maskmem_pos_enc = self._encode_new_memory(
                current_vision_feats=ego_current_vision_feats,
                feat_sizes=feat_sizes,
                pred_masks_high_res=ego_gt_masks,
                object_score_logits=torch.tensor([[1.0]]).to("cuda"),
                is_mask_from_pts=False,
            )
            ego_current_out["maskmem_features"] = ego_maskmem_features
            ego_current_out["maskmem_pos_enc"] = ego_maskmem_pos_enc

        else:
            exo_current_out["maskmem_features"] = None
            exo_current_out["maskmem_pos_enc"] = None
            ego_current_out["maskmem_features"] = None
            ego_current_out["maskmem_pos_enc"] = None

    def track_step(
        self,
        frame_idx,
        is_init_cond_frame,
        ego_current_vision_feats,
        exo_current_vision_feats,
        ego_current_vision_pos_embeds,
        exo_current_vision_pos_embeds,
        feat_sizes,
        ego_point_inputs, # None
        exo_point_inputs, # None
        ego_mask_inputs,  # None
        exo_mask_inputs,  # None
        ego_output_dict,  # {'cond_frame_outputs':{}, 'non_cond_frame_outputs':{}}
        exo_output_dict,  # {'cond_frame_outputs':{}, 'non_cond_frame_outputs':{}}
        num_frames,
        track_in_reverse=False,  # tracking in reverse time order (for demo usage)
        # Whether to run the memory encoder on the predicted masks. Sometimes we might want
        # to skip the memory encoder with `run_mem_encoder=False`. For example,
        # in demo we might call `track_step` multiple times for each user click,
        # and only encode the memory when the user finalizes their clicks. And in ablation
        # settings like SAM training on static images, we don't need the memory encoder.
        run_mem_encoder=True,
        # The previously predicted SAM mask logits (which can be fed together with new clicks in demo).
        prev_sam_mask_logits=None,
        ego_gt_masks=None,
        exo_gt_masks=None,
        ego_memories=None,
        exo_memories=None,
        ego_memories_pos=None,
        exo_memories_pos=None,
    ):
        exo_current_out, ego_current_out, sam_outputs, exo_high_res_features, exo_pix_feat, historical_feature, ego_memories, exo_memories, ego_memories_pos, exo_memories_pos = self._track_step(
            frame_idx,
            is_init_cond_frame,
            ego_current_vision_feats,
            exo_current_vision_feats,
            ego_current_vision_pos_embeds,
            exo_current_vision_pos_embeds,
            feat_sizes,
            ego_point_inputs,
            exo_point_inputs,
            ego_mask_inputs,
            exo_mask_inputs,
            ego_output_dict,
            exo_output_dict,
            num_frames,
            track_in_reverse,
            prev_sam_mask_logits,
            ego_gt_masks,
            exo_gt_masks,
            ego_memories,
            exo_memories,
            ego_memories_pos,
            exo_memories_pos
        )

        (
            _,
            _,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
            upscaled_embedding,
            src,
            src_teacher,
            res
        ) = sam_outputs

        exo_current_out["pred_masks"] = low_res_masks
        exo_current_out["pred_masks_high_res"] = high_res_masks
        exo_current_out["obj_ptr"] = obj_ptr
        exo_current_out["iou"] = ious
        if historical_feature is not None:
            exo_current_out["historical_feature"] = historical_feature
        if not self.training:
            # Only add this in inference (to avoid unused param in activation checkpointing;
            # it's mainly used in the demo to encode spatial memories w/ consolidated masks)
            exo_current_out["object_score_logits"] = object_score_logits
        exo_current_out["object_score_logits"] = object_score_logits

        self._encode_memory_in_output(
            exo_current_vision_feats,
            feat_sizes,
            exo_point_inputs,
            run_mem_encoder,
            high_res_masks,
            object_score_logits,
            exo_current_out,
            ego_current_out,
            ego_gt_masks,
            ego_current_vision_feats
        )
        return exo_current_out, ego_current_out, ego_memories, exo_memories, ego_memories_pos, exo_memories_pos


    def _use_multimask(self, is_init_cond_frame, point_inputs):
        """Whether to use multimask output in the SAM head."""
        num_pts = 0 if point_inputs is None else point_inputs["point_labels"].size(1)
        multimask_output = (
            self.multimask_output_in_sam # True
            and (is_init_cond_frame or self.multimask_output_for_tracking)
            and (self.multimask_min_pt_num <= num_pts <= self.multimask_max_pt_num) # 为什么之前是false呢？
        )#  multimask_min_pt_num=1, multimask_max_pt_num=1,
        return multimask_output

    def _apply_non_overlapping_constraints(self, pred_masks):
        """
        Apply non-overlapping constraints to the object scores in pred_masks. Here we
        keep only the highest scoring object at each spatial location in pred_masks.
        """
        batch_size = pred_masks.size(0)
        if batch_size == 1:
            return pred_masks

        device = pred_masks.device
        # "max_obj_inds": object index of the object with the highest score at each location
        max_obj_inds = torch.argmax(pred_masks, dim=0, keepdim=True)
        # "batch_obj_inds": object index of each object slice (along dim 0) in `pred_masks`
        batch_obj_inds = torch.arange(batch_size, device=device)[:, None, None, None]
        keep = max_obj_inds == batch_obj_inds
        # suppress overlapping regions' scores below -10.0 so that the foreground regions
        # don't overlap (here sigmoid(-10.0)=4.5398e-05)
        pred_masks = torch.where(keep, pred_masks, torch.clamp(pred_masks, max=-10.0))
        return pred_masks


def memory_bank_compress(memory_bank: torch.Tensor, memory_pos: torch.Tensor):
    """
    Compresses the memory bank if the current memory bank length is greater than the threshold.
    Compression_size is the number of frames that are compressed into each position.

    Args:
        memory_bank (torch.Tensor): The input memory bank to compress. Shape: (B, T, N, C)
        compression_size (torch.Tensor): The number of frames to compress into each position. Shape: (B, T, N)

    Returns:
        compressed_memory_bank (torch.Tensor): The compressed memory bank. Shape: (B, T-1, N, C)
        compressed_size (torch.Tensor): The number of frames compressed into each position. Shape: (B, T-1, N)
    """
    memory_bank = memory_bank.permute(2, 0, 1, 3)
    memory_pos = memory_pos.permute(2, 0, 1, 3)
    B, T, N, C = memory_bank.shape
    compression_size = torch.ones(B, T, N).to(memory_bank.device)
    # Calculate the cosine similarity between adjacent frames
    # similarity_matrix = F.cosine_similarity(memory_bank[:, :-1, :], memory_bank[:, 1:, :], dim=-1)
    # Select the frame indices with the top-1 similarity
    # _, max_similarity_indices = torch.max(similarity_matrix, dim=1, keepdim=True)
    similarity_matrix = torch.norm(memory_bank[:, :-1, :] - memory_bank[:, 1:, :], dim=-1, p=2)
    # Select the frame indices with the top-1 similarity
    _, max_similarity_indices = torch.min(similarity_matrix, dim=1, keepdim=True)

    # Calculate source and dst indices for compression
    src_indices = max_similarity_indices + 1
    dst_indices = torch.arange(T - 1).to(memory_bank.device)[None, :, None].repeat(B, 1, N)

    dst_indices[dst_indices > max_similarity_indices] += 1

    # Gather source and dst memory banks and sizes
    src_memory_bank = memory_bank.gather(dim=1, index=src_indices.unsqueeze(-1).expand(-1, -1, -1, C))
    src_memory_pos = memory_pos.gather(dim=1, index=src_indices.unsqueeze(-1).expand(-1, -1, -1, C))
    dst_memory_bank = memory_bank.gather(dim=1, index=dst_indices.unsqueeze(-1).expand(-1, -1, -1, C))
    dst_memory_pos = memory_pos.gather(dim=1, index=dst_indices.unsqueeze(-1).expand(-1, -1, -1, C))
    src_size = compression_size.gather(dim=1, index=src_indices)
    dst_size = compression_size.gather(dim=1, index=dst_indices)

    # Multiply the memory banks by their corresponding sizes
    src_memory_bank *= src_size.unsqueeze(-1)
    dst_memory_bank *= dst_size.unsqueeze(-1)
    src_memory_pos *= src_size.unsqueeze(-1)
    dst_memory_pos *= dst_size.unsqueeze(-1)
    # Compress the memory bank by adding the source memory bank to the dst memory bank
    dst_memory_bank.scatter_add(dim=1, index=max_similarity_indices.unsqueeze(-1).expand(-1, -1, -1, C),
                                 src=src_memory_bank)
    dst_memory_pos.scatter_add(dim=1, index=max_similarity_indices.unsqueeze(-1).expand(-1, -1, -1, C),
                                 src=src_memory_pos)

    dst_size.scatter_add(dim=1, index=max_similarity_indices, src=src_size)

    # Normalize the dst memory bank by its size
    compressed_memory_bank = dst_memory_bank / dst_size.unsqueeze(-1)
    compressed_memory_pos = dst_memory_pos / dst_size.unsqueeze(-1)
    return compressed_memory_bank, compressed_memory_pos


def kmeans_cluster_frame(batch_features, K):

    B, N, C = batch_features.shape
    batch_keyframes = []

    for b in range(B):
        frame_vectors = batch_features[b]
        frame_vectors = np.asarray(frame_vectors.cpu().to(torch.float32))


        K_b = min(K, N)

        kmeans = KMeans(n_clusters=K_b, random_state=42, init='k-means++', n_init='auto')
        clusters = kmeans.fit_predict(frame_vectors)
        centroids = kmeans.cluster_centers_


        keyframes = []
        for i in range(K_b):
            cluster_indices = np.where(clusters == i)[0]
            distances = np.linalg.norm(frame_vectors[cluster_indices] - centroids[i], axis=1)  # 计算与中心距离
            keyframe_idx = cluster_indices[np.argmin(distances)]
            keyframes.append(keyframe_idx)
        keyframes.sort()
        batch_keyframes.append(keyframes)

    return batch_keyframes


class MV_MoE(nn.Module):
    def __init__(self, in_channels):
        super(MV_MoE, self).__init__()

        self.mlp1 = nn.Sequential(
            nn.Linear(in_channels*2, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, in_channels),
            nn.Sigmoid()
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(in_channels*2, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, in_channels),
            nn.Sigmoid()
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(2*in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(2*in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, memory_fused, view_guided):

        gap_memory_fused = torch.mean(memory_fused, dim=(2, 3))  # Global Average Pooling
        gap_view_guided = torch.mean(view_guided, dim=(2, 3))  # Global Average Pooling
        both_concat = torch.cat((gap_memory_fused, gap_view_guided), dim=1)
        w_memory = self.mlp1(both_concat).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        w_view = self.mlp2(both_concat).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]

        memory_fused = memory_fused + memory_fused * w_memory
        view_guided = view_guided + view_guided * w_view

        feature_concat = torch.cat((memory_fused, view_guided), dim=1)
        w_memory1 = self.conv1(feature_concat)
        w_view1 = self.conv2(feature_concat)

        memory_fused = memory_fused + memory_fused * w_memory1
        view_guided = view_guided + view_guided * w_view1
        out = memory_fused + view_guided

        return out