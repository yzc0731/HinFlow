import os

import hydra
import torch
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import lightning
from lightning.fabric import Fabric

from atm.model import *
from atm.dataloader import ATMPretrainDataset, get_dataloader
from atm.utils.log_utils import BestAvgLoss, MetricLogger
from atm.utils.train_utils import init_wandb, setup_lr_scheduler, setup_optimizer

@hydra.main(config_path="../conf/train_planner", version_base="1.3")
def main(cfg: DictConfig):
    work_dir = HydraConfig.get().runtime.output_dir
    print(f"work_dir: {work_dir}")
    setup(cfg)
    OmegaConf.save(config=cfg, f=os.path.join(work_dir, "config.yaml"))
    with open(f"{work_dir}/loss.log", "w") as f:
        f.write("")
    
    fabric = Fabric(accelerator="cuda", devices=list(cfg.train_gpus), precision="bf16-mixed" if cfg.mix_precision else None, strategy="deepspeed")
    fabric.launch()

    None if (cfg.dry or not fabric.is_global_zero) else init_wandb(cfg)

    train_dataset = ATMPretrainDataset(dataset_dir=cfg.train_dataset, **cfg.dataset_cfg, aug_prob=cfg.aug_prob)
    train_loader = get_dataloader(train_dataset, mode="train", num_workers=cfg.num_workers, batch_size=cfg.batch_size)

    val_dataset = ATMPretrainDataset(dataset_dir=cfg.val_dataset, **cfg.dataset_cfg, aug_prob=0.)
    val_loader = get_dataloader(val_dataset, mode="val", num_workers=cfg.num_workers, batch_size=cfg.batch_size)

    model_cls = eval(cfg.model_name)
    model = model_cls(**cfg.model_cfg)
    optimizer = setup_optimizer(cfg.optimizer_cfg, model)
    scheduler = setup_lr_scheduler(optimizer, cfg.scheduler_cfg)

    model, optimizer = fabric.setup(model, optimizer)
    model.mark_forward_method("forward_loss")
    train_loader = fabric.setup_dataloaders(train_loader)

    lbd_track = cfg.lbd_track
    lbd_img = cfg.lbd_img
    p_img = cfg.p_img

    # Pick ckpt based on  the average of the last 5 epochs
    metric_logger = MetricLogger(delimiter=" ")
    best_loss_logger = BestAvgLoss(window_size=5)

    for epoch in metric_logger.log_every(range(cfg.epochs), 1, ""):
        train_metrics = run_one_epoch(
            fabric,
            model,
            train_loader,
            optimizer,
            lbd_track=lbd_track,
            lbd_img=lbd_img,
            p_img=p_img,
            scheduler=scheduler,
            mix_precision=cfg.mix_precision,
            clip_grad=cfg.clip_grad,
        )

        train_metrics["train/lr"] = optimizer.param_groups[0]["lr"]
        metric_logger.update(**train_metrics)

        if fabric.is_global_zero:
            None if cfg.dry else wandb.log(train_metrics, step=epoch)

            if epoch % cfg.val_freq == 0:
                val_metrics = evaluate(
                    model,
                    val_loader,
                    lbd_track=lbd_track,
                    lbd_img=lbd_img,
                    p_img=p_img,
                    mix_precision=cfg.mix_precision,
                    tag="val",
                )

                # Save best checkpoint
                metric_logger.update(**val_metrics)

                val_metrics = {**val_metrics}
                loss_metric = val_metrics["val/loss"]
                with open(f"{work_dir}/loss.log", "a") as f:
                    f.write(f"Epoch: {epoch},")
                    # for loop over all item in train_metrics and val_metrics
                    for key, value in train_metrics.items():
                        f.write(f"{key.replace('/', '_')}: {value:.6f},")
                    for key, value in val_metrics.items():
                        f.write(f"{key.replace('/', '_')}: {value:.6f},")
                    f.write("\n")

                is_best = best_loss_logger.update_best(loss_metric, epoch)

                if is_best:
                    model.save(f"{work_dir}/model_best.ckpt")
                    with open(f"{work_dir}/best_epoch.txt", "a") as f:
                        f.write(
                            "Best epoch: %d, Best %s: %.7f"
                            % (epoch, "loss", best_loss_logger.best_loss)
                        )
                None if cfg.dry else wandb.log(val_metrics, step=epoch)

            if epoch % cfg.save_freq == 0:
                model.save(f"{work_dir}/model_{epoch}.ckpt")

    if fabric.is_global_zero:
        model.save(f"{work_dir}/model_final.ckpt")
        None if cfg.dry else print(f"finished training in {wandb.run.dir}")
        None if cfg.dry else wandb.finish()


def run_one_epoch(fabric,
                  model,
                  dataloader,
                  optimizer,
                  lbd_track,
                  lbd_img,
                  p_img,
                  mix_precision=False,
                  scheduler=None,
                  clip_grad=1.0):
    """
    Optimize the policy. Return a dictionary of the loss and any other metrics.
    """
    track_loss, vid_loss, tot_loss, tot_items = 0, 0, 0, 0
    agentview_track_loss, eyeinhand_track_loss = 0, 0

    model.train()
    i = 0
    for vid, track, vis, actions, task_emb, view_idx in tqdm(dataloader):
        if mix_precision:
            vid, track, vis, actions, task_emb = vid.bfloat16(), track.bfloat16(), vis.bfloat16(), actions.bfloat16(), task_emb.bfloat16()
        b, t, c, h, w = vid.shape
        b, tl, n, _ = track.shape
        b, tl, n = vis.shape
        loss, ret_dict = model.forward_loss(
            vid,
            track,
            actions,
            task_emb,
            lbd_track=lbd_track,
            lbd_img=lbd_img,
            view_idx=view_idx,
            p_img=p_img)  # do not use vis
        optimizer.zero_grad()
        fabric.backward(loss)

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        optimizer.step()

        track_loss += ret_dict["track_loss"]
        vid_loss += ret_dict["img_loss"]
        tot_loss += ret_dict["loss"]
        agentview_track_loss += ret_dict["agentview_track_loss"]
        eyeinhand_track_loss += ret_dict["eyeinhand_track_loss"]
        tot_items += b

        i += 1

    out_dict = {
        "train/track_loss": track_loss / tot_items,
        "train/vid_loss": vid_loss / tot_items,
        "train/loss": tot_loss / tot_items,
        "train/agentview_track_loss": agentview_track_loss / tot_items,
        "train/eyeinhand_track_loss": eyeinhand_track_loss / tot_items,
    }

    if scheduler is not None:
        scheduler.step()

    return out_dict

@torch.no_grad()
def evaluate(model, dataloader, lbd_track, lbd_img, p_img, mix_precision=False, tag="val"):
    track_loss, vid_loss, tot_loss, tot_items = 0, 0, 0, 0
    agentview_track_loss, eyeinhand_track_loss = 0, 0
    model.eval()

    i = 0
    for vid, track, vis, actions, task_emb, view_idx in tqdm(dataloader):
        vid, track, vis, actions, task_emb = vid.cuda(), track.cuda(), vis.cuda(), actions.cuda(), task_emb.cuda()
        if mix_precision:
            vid, track, vis, actions, task_emb = vid.bfloat16(), track.bfloat16(), vis.bfloat16(), actions.bfloat16(), task_emb.bfloat16()
        b, t, c, h, w = vid.shape
        b, tl, n, _ = track.shape

        _, ret_dict = model.forward_loss(
            vid,
            track,
            actions,
            task_emb,
            lbd_track=lbd_track,
            lbd_img=lbd_img,
            view_idx=view_idx,
            p_img=p_img,
            vis=vis)

        track_loss += ret_dict["track_loss"]
        vid_loss += ret_dict["img_loss"]
        tot_loss += ret_dict["loss"]
        agentview_track_loss += ret_dict["agentview_track_loss"]
        eyeinhand_track_loss += ret_dict["eyeinhand_track_loss"]
        tot_items += b

        i += 1

    out_dict = {
        f"{tag}/total_items": tot_items,
        f"{tag}/track_loss": track_loss / tot_items,
        f"{tag}/vid_loss": vid_loss / tot_items,
        f"{tag}/loss": tot_loss / tot_items,
        f"{tag}/agentview_track_loss": agentview_track_loss / tot_items, 
        f"{tag}/eyeinhand_track_loss": eyeinhand_track_loss / tot_items, 
    }

    return out_dict


def setup(cfg):
    import warnings

    warnings.simplefilter("ignore")

    lightning.seed_everything(cfg.seed)


if __name__ == "__main__":
    main()
