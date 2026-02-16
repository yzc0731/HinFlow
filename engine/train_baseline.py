import hydra
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.distributed as dist
import lightning
from lightning.fabric import Fabric

import os
import wandb
import json
from tqdm import tqdm
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from atm.dataloader import BCDataset, get_dataloader
from atm.policy import *
from atm.sampler import *
from atm.utils.train_utils import setup_optimizer, setup_lr_scheduler, init_wandb
from atm.utils.log_utils import MetricLogger, BestAvgLoss

from hydra.core.global_hydra import GlobalHydra
if GlobalHydra().is_initialized():
    GlobalHydra().clear()

@hydra.main(config_path="../conf/train_baseline", version_base="1.3")
def main(cfg: DictConfig):
    # Put the import here so that running on slurm does not have import error
    work_dir = HydraConfig.get().runtime.output_dir
    setup(cfg)
    OmegaConf.save(config=cfg, f=os.path.join(work_dir, "config.yaml"))
    # new a log file to record the loss
    with open(f"{work_dir}/loss.log", "w") as f:
        f.write("")

    train_dataset = BCDataset(dataset_dir=cfg.train_dataset, **cfg.dataset_cfg, aug_prob=cfg.aug_prob)
    train_loader = get_dataloader(train_dataset, mode="train", num_workers=cfg.num_workers, batch_size=cfg.batch_size)

    val_dataset = BCDataset(dataset_dir=cfg.val_dataset, **cfg.dataset_cfg, aug_prob=0.)
    val_loader = get_dataloader(val_dataset, mode="val", num_workers=cfg.num_workers, batch_size=cfg.batch_size)

    fabric = Fabric(accelerator="cuda", devices=list(cfg.train_gpus), precision="bf16-mixed" if cfg.mix_precision else None, strategy="deepspeed")
    fabric.launch()

    None if (cfg.dry or not fabric.is_global_zero) else init_wandb(cfg)

    model_cls = eval(cfg.model_name)
    model = model_cls(**cfg.model_cfg)
    optimizer = setup_optimizer(cfg.optimizer_cfg, model)
    scheduler = setup_lr_scheduler(optimizer, cfg.scheduler_cfg)

    fabric.barrier()
    model, optimizer = fabric.setup(model, optimizer)
    train_loader = fabric.setup_dataloaders(train_loader)

    # Pick ckpt based on  the average of the last 5 epochs
    metric_logger = MetricLogger(delimiter=" ")
    best_loss_logger = BestAvgLoss(window_size=5)

    fabric.barrier()
    for epoch in metric_logger.log_every(range(cfg.epochs), 1, ""):
        model.mark_forward_method("forward_loss")
        train_metrics = run_one_epoch(
            fabric,
            model,
            train_loader,
            optimizer,
            cfg.clip_grad,
            mix_precision=cfg.mix_precision,
            scheduler=scheduler,
        )

        train_metrics["train/lr"] = optimizer.param_groups[0]["lr"]
        metric_logger.update(**train_metrics)

        if fabric.is_global_zero:
            None if cfg.dry else wandb.log(train_metrics, step=epoch)

            if epoch % cfg.val_freq == 0:
                model.mark_forward_method("forward_val")
                val_metrics = evaluate(
                    model,
                    val_loader,
                    mix_precision=cfg.mix_precision,
                    tag="val"
                )

                # Save best checkpoint
                metric_logger.update(**val_metrics)

                val_metrics = {**val_metrics}
                loss_metric = val_metrics["val/loss"]
                is_best = best_loss_logger.update_best(loss_metric, epoch)
                # save val_metrics["val/loss"] to work_dir/loss.log
                with open(f"{work_dir}/loss.log", "a") as f:
                    f.write(f"Epoch: {epoch},")
                    # for loop over all item in train_metrics and val_metrics
                    for key, value in train_metrics.items():
                        f.write(f"{key.replace('/', '_')}: {value:.6f},")
                    for key, value in val_metrics.items():
                        f.write(f"{key.replace('/', '_')}: {value:.6f},")
                    f.write("\n")

                if is_best:
                    model.save(f"{work_dir}/model_best.ckpt")
                    with open(f"{work_dir}/best_epoch.txt", "a") as f:
                        f.write(
                            "Best epoch: %d, Best %s: %.4f"
                            % (epoch, "loss", best_loss_logger.best_loss)
                        )
                None if cfg.dry else wandb.log(val_metrics, step=epoch)

        if epoch % cfg.save_freq == 0:
            model.save(f"{work_dir}/model_{epoch}.ckpt")
        fabric.barrier()

    if fabric.is_global_zero:
        model.save(f"{work_dir}/model_final.ckpt")
        None if cfg.dry else print(f"finished training in {wandb.run.dir}")
        None if cfg.dry else wandb.finish()

def run_one_epoch(fabric,
                  model,
                  dataloader,
                  optimizer,
                  clip_grad=1.0,
                  mix_precision=False,
                  scheduler=None,
                  ):
    """
    Optimize the policy. Return a dictionary of the loss and any other metrics.
    """
    tot_loss_dict, tot_items = {}, 0

    model.train()
    i = 0
    for obs, track_obs, track, task_emb, action, extra_states, demo_id, sample_points in tqdm(dataloader):
        if mix_precision:
            obs, track_obs, track, task_emb, action = obs.bfloat16(), track_obs.bfloat16(), track.bfloat16(), task_emb.bfloat16(), action.bfloat16()
            extra_states = {k: v.bfloat16() for k, v in extra_states.items()}

        loss, ret_dict = model.forward_loss(obs, track_obs, track, task_emb, extra_states, action, sample_points)
        optimizer.zero_grad()
        fabric.backward(loss)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

        optimizer.step()

        for k, v in ret_dict.items():
            if k not in tot_loss_dict:
                tot_loss_dict[k] = 0
            tot_loss_dict[k] += v
        tot_items += 1

        i += 1

    out_dict = {}
    out_dict["train/tot_items"] = tot_items
    for k, v in tot_loss_dict.items():
        out_dict[f"train/{k}"] = tot_loss_dict[f"{k}"] / tot_items

    if scheduler is not None:
        scheduler.step()

    return out_dict


@torch.no_grad()
def evaluate(model, dataloader, mix_precision=False, tag="val"):
    tot_loss_dict, tot_items = {}, 0
    model.eval()

    i = 0
    for obs, track_obs, track, task_emb, action, extra_states, demo_id, sample_points in tqdm(dataloader):
        obs, track_obs, track, task_emb, action = obs.cuda(), track_obs.cuda(), track.cuda(), task_emb.cuda(), action.cuda()
        sample_points = sample_points.cuda()
        extra_states = {k: v.cuda() for k, v in extra_states.items()}
        if mix_precision:
            obs, track_obs, track, task_emb, action = obs.bfloat16(), track_obs.bfloat16(), track.bfloat16(), task_emb.bfloat16(), action.bfloat16()
            extra_states = {k: v.bfloat16() for k, v in extra_states.items()}

        _, ret_dict = model.forward_val(obs, track_obs, track, task_emb, extra_states, action, sample_points)

        i += 1

        for k, v in ret_dict.items():
            if k not in tot_loss_dict:
                tot_loss_dict[k] = 0
            tot_loss_dict[k] += v
        tot_items += 1

    out_dict = {}
    out_dict[f"{tag}/tot_items"] = tot_items
    for k, v in tot_loss_dict.items():
        out_dict[f"{tag}/{k}"] = tot_loss_dict[f"{k}"] / tot_items

    return out_dict


def setup(cfg):
    import warnings

    warnings.simplefilter("ignore")

    lightning.seed_everything(cfg.seed)


if __name__ == "__main__":
    main()