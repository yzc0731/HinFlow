import numpy as np
from collections import deque
import robomimic.utils.tensor_utils as TensorUtils
from omegaconf import OmegaConf
import torch
import torch.nn as nn
import torchvision.transforms as T
import os
from einops import rearrange, repeat

from atm.model import *
from atm.model.track_patch_embed import TrackPatchEmbed
from atm.policy.vilt_modules.transformer_modules import *
from atm.policy.vilt_modules.rgb_modules import *
from atm.policy.vilt_modules.language_modules import *
from atm.policy.vilt_modules.extra_state_modules import ExtraModalityTokens
from atm.policy.vilt_modules.policy_head import *
from atm.utils.flow_utils import ImageUnNormalize

###############################################################################
#
# A ViLT Policy
#
###############################################################################


class BCViLTPolicy(nn.Module):
    """
    Input: (o_{t-H}, ... , o_t)
    Output: a_t or distribution of a_t
    """

    def __init__(self, 
                 obs_cfg, 
                 img_encoder_cfg, 
                 language_encoder_cfg, 
                 extra_state_encoder_cfg, 
                 track_cfg,
                 spatial_transformer_cfg, 
                 temporal_transformer_cfg,
                 policy_head_cfg, 
                 robot,
                 use_early_fusion=False,
                 use_late_fusion=True,
                 action_chunk=None,
                 load_path=None):
        super().__init__()
        self.use_early_fusion = use_early_fusion
        self.use_late_fusion = use_late_fusion

        self._process_obs_shapes(**obs_cfg)

        # 1. encode image
        self._setup_image_encoder(**img_encoder_cfg)

        # 2. encode language (spatial)
        self.language_encoder_spatial = self._setup_language_encoder(output_size=self.spatial_embed_size, **language_encoder_cfg)

        # 3. Track Transformer module
        self._setup_track(**track_cfg)

        # 3. define spatial positional embeddings, modality embeddings, and spatial token for summary
        self._setup_spatial_positional_embeddings()

        # 4. define spatial transformer
        self._setup_spatial_transformer(**spatial_transformer_cfg)

        ### 5. encode extra information (e.g. gripper, joint_state)
        self.extra_encoder = self._setup_extra_state_encoder(robot=robot, extra_embedding_size=self.temporal_embed_size, **extra_state_encoder_cfg)

        # 6. encode language (temporal), this will also act as the TEMPORAL_TOKEN, i.e., CLS token for action prediction
        self.language_encoder_temporal = self._setup_language_encoder(output_size=self.temporal_embed_size, **language_encoder_cfg)

        # 7. define temporal transformer
        self._setup_temporal_transformer(**temporal_transformer_cfg)

        # 8. define policy head
        self.action_chunk = action_chunk
        self.action_history = []
        self._setup_policy_head(**policy_head_cfg)

        if load_path is not None and os.path.exists(load_path):
            print("[INFO] load planner ckpt from:", load_path)
            self.load(load_path)
            self.track.load(f"{track_cfg.track_fn}/model_best.ckpt")
        else:
            print("[INFO] planner ckpt not exist:", load_path)

    def _process_obs_shapes(self, obs_shapes, num_views, extra_states, img_mean, img_std, max_seq_len):
        self.img_normalizer = T.Normalize(img_mean, img_std)
        self.img_unnormalizer = ImageUnNormalize(img_mean, img_std)
        self.obs_shapes = obs_shapes
        self.img_shapes = obs_shapes['rgb']
        self.policy_num_track_ts = obs_shapes["tracks"][0]
        self.policy_num_track_ids = obs_shapes["tracks"][1]
        self.num_views = num_views
        self.extra_state_keys = extra_states
        self.max_seq_len = max_seq_len
        # define buffer queue for encoded latent features
        self.latent_queue = deque(maxlen=max_seq_len)
        self.track_obs_queue = deque(maxlen=max_seq_len)

    def _setup_image_encoder(self, network_name, patch_size, embed_size, no_patch_embed_bias):
        self.spatial_embed_size = embed_size
        self.image_encoders = []
        for _ in range(self.num_views):
            input_shape = self.obs_shapes["rgb"]
            self.image_encoders.append(eval(network_name)(input_shape=input_shape, patch_size=patch_size,
                                                          embed_size=self.spatial_embed_size,
                                                          no_patch_embed_bias=no_patch_embed_bias))
        self.image_encoders = nn.ModuleList(self.image_encoders)

        self.img_num_patches = sum([x.num_patches for x in self.image_encoders])

    def _setup_language_encoder(self, network_name, **language_encoder_kwargs):
        return eval(network_name)(**language_encoder_kwargs)

    def _setup_track(self, track_fn, policy_track_patch_size=None, use_zero_track=False, use_ground_truth_track=False):
        """
        track_fn: path to the track model
        policy_track_patch_size: The patch size of TrackPatchEmbedding in the policy, if None, it will be assigned the same patch size as TrackTransformer by default
        use_zero_track: whether to zero out the tracks (ie use only the image)
        use_ground_truth_track: use the ground truth track instead of the predicted track
        """

        cfg_path = os.path.join(track_fn, "config.yaml")
        track_cfg = OmegaConf.load(cfg_path)
        self.use_zero_track = use_zero_track
        self.use_ground_truth_track = use_ground_truth_track

        track_model_path = os.path.join(track_fn, "model_best.ckpt")
        track_cfg.model_cfg.load_path = track_model_path
        track_cls = eval(track_cfg.model_name)
        self.track = track_cls(**track_cfg.model_cfg)

        # freeze
        self.track.eval()
        for param in self.track.parameters():
            param.requires_grad = False

        self.num_track_ids = self.track.num_track_ids
        self.num_track_ts = self.track.num_track_ts
        self.policy_track_patch_size = self.track.track_patch_size if policy_track_patch_size is None else policy_track_patch_size
        print("[ViLT] num_track_ids num_track_ts policy_track_patch_size", self.num_track_ids, self.num_track_ts, self.policy_track_patch_size)


        self.track_proj_encoder = TrackPatchEmbed(
            num_track_ts=self.policy_num_track_ts,
            num_track_ids=self.num_track_ids,
            patch_size=self.policy_track_patch_size,
            in_dim=2 + self.num_views,  # X, Y, one-hot view embedding
            embed_dim=self.spatial_embed_size)

        self.track_id_embed_dim = 16
        self.num_track_patches_per_view = self.track_proj_encoder.num_patches_per_track
        self.num_track_patches = self.num_track_patches_per_view * self.num_views

    def _setup_spatial_positional_embeddings(self):
        # setup positional embeddings
        spatial_token = nn.Parameter(torch.randn(1, 1, self.spatial_embed_size))  # SPATIAL_TOKEN
        img_patch_pos_embed = nn.Parameter(torch.randn(1, self.img_num_patches, self.spatial_embed_size))

        track_patch_pos_embed = nn.Parameter(torch.randn(1, self.num_track_patches, self.spatial_embed_size))

        modality_embed = nn.Parameter(
            torch.randn(1, len(self.image_encoders) + self.num_views + 1, self.spatial_embed_size)
        )  # IMG_PATCH_TOKENS + TRACK_PATCH_TOKENS + SENTENCE_TOKEN

        self.register_parameter("spatial_token", spatial_token)
        self.register_parameter("img_patch_pos_embed", img_patch_pos_embed)
        self.register_parameter("track_patch_pos_embed", track_patch_pos_embed)
        self.register_parameter("modality_embed", modality_embed)

        # for selecting modality embed
        modality_idx = []
        for i, encoder in enumerate(self.image_encoders):
            modality_idx += [i] * encoder.num_patches
        
        if self.use_early_fusion:
            modality_idx += [modality_idx[-1] + 1 + i for i in range(self.num_views)] * self.num_track_ids * self.num_track_patches_per_view
        
        modality_idx += [modality_idx[-1] + 1]  # for sentence embedding
        self.modality_idx = torch.LongTensor(modality_idx)

    def _setup_extra_state_encoder(self, **extra_state_encoder_cfg):
        if len(self.extra_state_keys) == 0:
            return None
        else:
            return ExtraModalityTokens(
                use_joint=("joint_states" in self.extra_state_keys),
                use_gripper=("gripper_states" in self.extra_state_keys),
                use_ee=("ee_states" in self.extra_state_keys),
                use_tcp=("tcp_pose" in self.extra_state_keys),
                **extra_state_encoder_cfg
            )

    def _setup_spatial_transformer(self, num_layers, num_heads, head_output_size, mlp_hidden_size, dropout,
                                   spatial_downsample, spatial_downsample_embed_size, use_language_token):
        self.spatial_transformer = TransformerDecoder(
            input_size=self.spatial_embed_size,
            num_layers=num_layers,
            num_heads=num_heads,
            head_output_size=head_output_size,
            mlp_hidden_size=mlp_hidden_size,
            dropout=dropout,
        )

        if spatial_downsample:
            self.temporal_embed_size = spatial_downsample_embed_size
            self.spatial_downsample = nn.Linear(self.spatial_embed_size, self.temporal_embed_size)
        else:
            self.temporal_embed_size = self.spatial_embed_size
            self.spatial_downsample = nn.Identity()

        self.spatial_transformer_use_text = use_language_token

    def _setup_temporal_transformer(self, num_layers, num_heads, head_output_size, mlp_hidden_size, dropout, use_language_token):
        self.temporal_position_encoding_fn = SinusoidalPositionEncoding(input_size=self.temporal_embed_size)

        self.temporal_transformer = TransformerDecoder(
            input_size=self.temporal_embed_size,
            num_layers=num_layers,
            num_heads=num_heads,
            head_output_size=head_output_size,
            mlp_hidden_size=mlp_hidden_size,
            dropout=dropout,)
        self.temporal_transformer_use_text = use_language_token

        action_cls_token = nn.Parameter(torch.zeros(1, 1, self.temporal_embed_size))
        nn.init.normal_(action_cls_token, std=1e-6)
        self.register_parameter("action_cls_token", action_cls_token)

    def _setup_policy_head(self, network_name, **policy_head_kwargs):
        policy_head_kwargs["input_size"] = self.temporal_embed_size
        if self.use_late_fusion:
            policy_head_kwargs["input_size"] += self.num_views * self.policy_num_track_ts * self.policy_num_track_ids * 2

        action_shape = policy_head_kwargs["output_size"]
        if self.action_chunk is not None:
            action_shape = [self.action_chunk] + list(action_shape)
        self.act_shape = action_shape
        self.out_shape = np.prod(action_shape)
        policy_head_kwargs["output_size"] = self.out_shape
        self.policy_head = eval(network_name)(**policy_head_kwargs)

    @torch.no_grad()
    def preprocess(self, obs, track, action):
        """
        Preprocess observations, according to an observation dictionary.
        Return the feature and state.
        """
        b, v, t, c, h, w = obs.shape

        action = action.reshape(b, t, self.out_shape)

        obs = self._preprocess_rgb(obs)

        return obs, track, action

    @torch.no_grad()
    def _preprocess_rgb(self, rgb):
        rgb = self.img_normalizer(rgb / 255.)
        return rgb

    def _get_view_one_hot(self, tr):
        """ tr: b, v, t, tl, n, d -> (b, v, t), tl n, d + v"""
        b, v, t, tl, n, d = tr.shape
        tr = rearrange(tr, "b v t tl n d -> (b t tl n) v d")
        one_hot = torch.eye(v, device=tr.device, dtype=tr.dtype)[None, :, :].repeat(tr.shape[0], 1, 1)
        tr_view = torch.cat([tr, one_hot], dim=-1)  # (b t tl n) v (d + v)
        tr_view = rearrange(tr_view, "(b t tl n) v c -> b v t tl n c", b=b, v=v, t=t, tl=tl, n=n, c=d + v)
        return tr_view

    def track_encode(self, track_obs, track, task_emb, track_points):
        """
        Args:
            track_obs: b v t tt_fs c h w
            track: shape (b v t track_len n 2). during the evaluation, the track is none, so it will use predicted track
            task_emb: b e
        Returns: b v t track_len n 2
        """
        b, v, t, *_ = track_obs.shape

        if self.use_ground_truth_track and track is not None:
            assert self.use_zero_track is False, "cannot use both ground truth track and zero track"
            recon_tr = track.clone()
        elif self.use_zero_track:
            recon_tr = torch.zeros((b, v, t, self.num_track_ts, self.num_track_ids, 2), device=track_obs.device, dtype=track_obs.dtype)
        else:
            track_obs_to_pred = rearrange(track_obs, "b v t fs c h w -> (b v t) fs c h w")
        
            if len(track_points.shape) == 2:
                # (n, d)
                track_points = repeat(track_points, "n d -> b v t n d", b=b, v=v, t=t)
            elif len(track_points.shape) == 3:
                # (1, n, d) or (b, n, d)
                if track_points.shape[0] == 1 and b > 1:
                    track_points = track_points.repeat(b, 1, 1) # (1, n, d) -> (b, n, d)
                track_points = repeat(track_points, "b n d -> b v t n d", v=v, t=t)
            elif len(track_points.shape) == 4:
                # (1, v, n, d) or (b, v, n, d)
                if track_points.shape[0] == 1 and b > 1:
                    track_points = track_points.repeat(b, 1, 1, 1) # (1, v, n, d) -> (b, v, n, d)
                track_points = repeat(track_points, "b v n d -> b v t n d", t=t)

            assert len(track_points.shape) == 5, f"track_points.shape: {track_points.shape}, track_obs.shape: {track_obs.shape}" # (b, v, t, n, 2)
            assert track_points.shape[:3] == track_obs.shape[:3], f"track_points.shape: {track_points.shape}, track_obs.shape: {track_obs.shape}"
            sampled_track = repeat(track_points, "b v t n d -> b v t tl n d", tl=self.num_track_ts) # (b, v, t, tl=16, n, d)
            sampled_track = rearrange(sampled_track, "b v t tl n d -> (b v t) tl n d")

            expand_task_emb = repeat(task_emb, "b e -> b v t e", b=b, v=v, t=t)
            expand_task_emb = rearrange(expand_task_emb, "b v t e -> (b v t) e")
            with torch.no_grad():
                pred_tr, _ = self.track.reconstruct(track_obs_to_pred, sampled_track, expand_task_emb, p_img=0)  # (b v t) tl n d
                recon_tr = rearrange(pred_tr, "(b v t) tl n d -> b v t tl n d", b=b, v=v, t=t)

        recon_tr = recon_tr[:, :, :, :self.policy_num_track_ts, :, :]  # truncate the track to a shorter one
        _recon_tr = recon_tr.clone()  # b v t tl n 2
        with torch.no_grad():
            tr_view = self._get_view_one_hot(recon_tr)  # b v t tl n c

        tr_view = rearrange(tr_view, "b v t tl n c -> (b v t) tl n c")
        tr = self.track_proj_encoder(tr_view)  # (b v t) track_patch_num n d
        tr = rearrange(tr, "(b v t) pn n d -> (b t n) (v pn) d", b=b, v=v, t=t, n=self.num_track_ids)  # (b t n) (v patch_num) d

        return tr, _recon_tr
    
    def spatial_encode(self, obs, track_obs, task_emb, extra_states, track_points, track_len=None, track=None, return_recon=False, online=False):
        """
        Encode the images separately in the videos along the spatial axis.
        Args:
            obs: b v t c h w
            track_obs: b v t tt_fs c h w, (0, 255)
            track: b v t track_len n 2
            task_emb: b e
            extra_states: {k: b t n}
        Returns: out: (b t 2+num_extra c), recon_track: (b v t tl n 2)
        """
        # 1. encode image
        img_encoded = []
        for view_idx in range(self.num_views):
            img_encoded.append(
                rearrange(
                    TensorUtils.time_distributed(
                        obs[:, view_idx, ...], self.image_encoders[view_idx], online=online
                    ),
                    "b t c h w -> b t (h w) c",
                )
            )  # (b, t, num_patches, c)

        img_encoded = torch.cat(img_encoded, -2)  # (b, t, 2*num_patches, c)
        img_encoded += self.img_patch_pos_embed.unsqueeze(0)  # (b, t, 2*num_patches, c)
        B, T = img_encoded.shape[:2]

        # 2. encode task_emb
        text_encoded = self.language_encoder_spatial(task_emb)  # (b, c)
        text_encoded = text_encoded.view(B, 1, 1, -1).expand(-1, T, -1, -1)  # (b, t, 1, c)

        # 3. encode track
        track_encoded, _recon_track = self.track_encode(track_obs, track, task_emb, track_points=track_points)  # track_encoded: ((b t n), 2*patch_num, c)  _recon_track: (b, v, track_len, n, 2)
        if track_len is None:
            track_len = torch.tensor([self.policy_num_track_ts-1], device=track_encoded.device, dtype=torch.long).repeat(B, T)  # (b, t)
        
        if self.use_early_fusion:
            # patch position embedding
            track_encoded += self.track_patch_pos_embed  # ((b t n), 2*patch_num, c)

            track_encoded = rearrange(track_encoded, "(b t n) pn d -> b t (n pn) d", b=B, t=T)  # (b, t, num_track*2*num_track_patch, c)

            # 3. concat img + track + text embs then add modality embeddings
            if self.spatial_transformer_use_text:
                img_track_text_encoded = torch.cat([img_encoded, track_encoded, text_encoded], -2)  # (b, t, 2*num_img_patch + num_track*2*num_track_patch + 1, c)
                img_track_text_encoded += self.modality_embed[None, :, self.modality_idx, :]
            else:
                img_track_text_encoded = torch.cat([img_encoded, track_encoded], -2)  # (b, t, 2*num_img_patch + 2*num_track*num_track_patch, c)

                # add modality embedding
                img_track_text_encoded += self.modality_embed[None, :, self.modality_idx[:-1], :]
        else:
            # only use late fusion
            if self.spatial_transformer_use_text:
                img_track_text_encoded = torch.cat([img_encoded, text_encoded], -2)
                img_track_text_encoded += self.modality_embed[None, :, self.modality_idx, :]
            else:
                img_track_text_encoded = img_encoded
                img_track_text_encoded += self.modality_embed[None, :, self.modality_idx[:-1], :]

        # 4. add spatial token
        spatial_token = self.spatial_token.unsqueeze(0).expand(B, T, -1, -1)  # (b, t, 1, c)
        encoded = torch.cat([spatial_token, img_track_text_encoded], -2)  # (b, t, 1+2*num_img_patch + num_track*2*num_track_patch, c)

        # 5. pass through transformer
        encoded = rearrange(encoded, "b t n c -> (b t) n c")  # (b*t, 1+2*num_img_patch + num_track*2*num_track_patch, c)
        out = self.spatial_transformer(encoded)
        out = out[:, 0]  # extract spatial token as summary at o_t
        out = self.spatial_downsample(out).view(B, T, 1, -1)  # (b, t, 1, c')

        # 6. encode extra states
        if self.extra_encoder is None:
            extra = None
        else:
            extra = self.extra_encoder(extra_states)  # (B, T, num_extra, c')

        # 7. encode language, treat it as action token
        action_cls_token = self.action_cls_token.unsqueeze(0).expand(B, T, -1, -1)  # (b, t, 1, c')
        if self.temporal_transformer_use_text:
            text_encoded_ = self.language_encoder_temporal(task_emb)  # (b, c')
            text_encoded_ = text_encoded_.view(B, 1, 1, -1).expand(-1, T, -1, -1)  # (b, t, 1, c')
            out_seq = [action_cls_token, text_encoded_, out]
        else:
            out_seq = [action_cls_token, out]

        if self.extra_encoder is not None:
            out_seq.append(extra)
        output = torch.cat(out_seq, -2)  # (b, t, 2 or 3 + num_extra, c')

        if return_recon:
            output = (output, _recon_track)

        return output

    def temporal_encode(self, x):
        """
        Args:
            x: b, t, num_modality, c
        Returns:
        """
        pos_emb = self.temporal_position_encoding_fn(x)  # (t, c)
        x = x + pos_emb.unsqueeze(1)  # (b, t, 2+num_extra, c)
        sh = x.shape
        assert self.temporal_transformer.mask is None

        x = TensorUtils.join_dimensions(x, 1, 2)  # (b, t*num_modality, c)
        x = self.temporal_transformer(x)
        x = x.reshape(*sh)  # (b, t, num_modality, c)
        return x[:, :, 0]  # (b, t, c)

    def forward(self, obs, track_obs, track, task_emb, extra_states, sample_points, track_len=None, online=False):
        """
        Return feature and info.
        Args:
            obs: b v t c h w
            track_obs: b v t tt_fs c h w
            track: b v t track_len n 2, used for training when use_ground_truth_track is True
            extra_states: {k: b t e}
        """
        x, recon_track = self.spatial_encode(obs, track_obs, task_emb, extra_states, track_len=track_len, track_points=sample_points, track=track, return_recon=True, online=online)  # x: (b, t, 2+num_extra, c), recon_track: (b, v, t, tl, n, 2)
        x = self.temporal_encode(x)  # (b, t, c)

        recon_track = rearrange(recon_track, "b v t tl n d -> b t (v tl n d)")
        if self.use_late_fusion:
            x = torch.cat([x, recon_track], dim=-1)  # (b, t, c + v*tl*n*2)

        dist = self.policy_head(x)  # only use the current timestep feature to predict action
        return dist

    def forward_common(self, obs, track_obs, track, task_emb, extra_states, action, sample_points, track_len=None, return_dist=False, online=False):
        """
        Common forward logic for both training and validation.
        Args:
            obs: b v t c h w
            track_obs: b v t tt_fs c h w
            track: b v t track_len n 2, used for training when use_ground_truth_track is True
            task_emb: b emb_size
            action: b t act_dim
        """
        obs, track, action = self.preprocess(obs, track, action)
        dist = self.forward(obs, track_obs, track, task_emb, extra_states, sample_points, track_len=track_len, online=online)
        loss = self.policy_head.loss_fn(dist, action, reduction="mean")
        
        ret_dict = {
            "bc_loss": loss.sum().item(),
        }

        if not self.policy_head.deterministic:
            # pseudo loss
            sampled_action = dist.sample().detach()
            mse_loss = F.mse_loss(sampled_action, action)
            ret_dict["pseudo_sampled_action_mse_loss"] = mse_loss.sum().item()

        ret_dict["loss"] = ret_dict["bc_loss"]
        if return_dist:
            return loss.sum(), ret_dict, (dist, action)
        return loss.sum(), ret_dict # only loss.sum() is backpropagated, any other thing is just for logging

    def forward_loss(self, obs, track_obs, track, task_emb, extra_states, action, sample_points, track_len=None, return_dist=False, online=False):
        """
        Forward logic for training.
        """
        assert self.training, "model is not in training mode"
        return self.forward_common(obs, track_obs, track, task_emb, extra_states, action, sample_points, track_len, return_dist, online=online)

    def forward_val(self, obs, track_obs, track, task_emb, extra_states, action, sample_points, track_len=None, return_dist=False):
        """
        Forward logic for validation.
        """
        return self.forward_common(obs, track_obs, track, task_emb, extra_states, action, sample_points, track_len, return_dist)

    def act(self, obs, task_emb, extra_states, track_points, track=None):
        """online agents will call this `act` function
        Args:
            obs: (b, v, h, w, c)
            task_emb: (b, em_dim)
            extra_states: {k: (b, state_dim,)}
        """
        training_mode = self.training
        self.eval()
        B = obs.shape[0]

        # expand time dimenstion
        obs = rearrange(obs, "b v h w c -> b v 1 c h w").copy()
        extra_states = {k: rearrange(v, "b e -> b 1 e") for k, v in extra_states.items()}

        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device
        # obs = torch.Tensor(obs).to(device=device, dtype=dtype)
        # use `as_tensor` rather than `torch.Tensor` is more resource efficient
        obs = torch.as_tensor(obs, device=device)
        obs = obs.float()
        
        task_emb = torch.Tensor(task_emb).to(device=device, dtype=dtype)
        extra_states = {k: torch.Tensor(v).to(device=device, dtype=dtype) for k, v in extra_states.items()}
        track_points = track_points.to(device=device, dtype=dtype)

        if (obs.shape[-2] != self.obs_shapes["rgb"][-2]) or (obs.shape[-1] != self.obs_shapes["rgb"][-1]):
            obs = rearrange(obs, "b v fs c h w -> (b v fs) c h w")
            obs = F.interpolate(obs, size=self.obs_shapes["rgb"][-2:], mode="bilinear", align_corners=False)
            obs = rearrange(obs, "(b v fs) c h w -> b v fs c h w", b=B, v=self.num_views)

        while len(self.track_obs_queue) < self.max_seq_len:
            self.track_obs_queue.append(torch.zeros_like(obs))
        self.track_obs_queue.append(obs.clone())
        track_obs = torch.cat(list(self.track_obs_queue), dim=2)  # b v fs c h w
        track_obs = rearrange(track_obs, "b v fs c h w -> b v 1 fs c h w")

        obs = self._preprocess_rgb(obs)

        with torch.no_grad():
            # step 1: calculate tracks
            x, rec_tracks = self.spatial_encode(obs, track_obs, task_emb=task_emb, extra_states=extra_states, return_recon=True, track_points=track_points, track=track)  # x: (b, 1, 4, c), recon_track: (b, v, 1, tl, n, 2)
            self.latent_queue.append(x)
            x = torch.cat(list(self.latent_queue), dim=1)  # (b, t, 4, c)
            # step 2 calculate latent
            x = self.temporal_encode(x)  # (b, t, c)

            if self.use_late_fusion:
                feat = torch.cat([x[:, -1], rearrange(rec_tracks[:, :, -1, :, :, :], "b v tl n d -> b (v tl n d)")], dim=-1)
            else:
                feat = x[:, -1]

            # calculate action
            action = self.policy_head.get_action(feat)  # only use the current timestep feature to predict action
            action = action.detach().cpu()  # (b, act_dim)

        action = action.reshape(-1, *self.act_shape)
        action = torch.clamp(action, -1, 1)
        self.train(training_mode)
        return action.float().cpu().numpy(), (None, rec_tracks[:, :, -1, :, :, :].cpu())  # (b, *act_shape)

    def act_atm_eval(self, obs, task_emb, extra_states, track_points, track=None):
        '''
        A wrapper function for atm evaluation with action chunking and exponential moving average
        '''
        action, _tracks = self.act(obs, task_emb, extra_states, track_points, track=track)
        self.action_history.append(action)
        final_action = action
        if self.action_chunk:
            action_slice = self.action_history[-self.action_chunk:] # action_chunk * (b, action_chunk, act_dim)
            Len_a = len(action_slice)
            k = 0.01
            exp_weights = np.exp(-k * np.arange(Len_a))
            exp_weights = exp_weights / exp_weights.sum()
            final_action = np.zeros_like(action[:, 0]) # b, 7
            for i in range(Len_a):
                final_action += exp_weights[i] * action_slice[i][:, Len_a - i - 1]
        return final_action, _tracks

    def reset(self):
        self.latent_queue.clear()
        self.track_obs_queue.clear()

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        weights = torch.load(path, map_location="cpu")
        for key in list(weights.keys()):
            if key.startswith("track."):
                del weights[key]
        self.load_state_dict(weights, strict=False)

    def train(self, mode=True):
        super().train(mode)
        self.track.eval()

    def eval(self):
        super().eval()
        self.track.eval()
