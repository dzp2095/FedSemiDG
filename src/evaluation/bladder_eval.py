# encoding: utf-8
import logging
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import torch 
from PIL import Image
from monai.metrics.meandice import DiceMetric
from monai.metrics.meaniou import MeanIoU
from monai.metrics.hausdorff_distance import HausdorffDistanceMetric
from monai.metrics.surface_distance import SurfaceDistanceMetric

from src.utils.draw import draw_mask_and_save
from src.utils.metric_logger import EMAMetricLogger
from src.evaluation.evaluation_strategy import EvaluationStrategy

class BladderEvalStrategy(EvaluationStrategy):
    def __init__(self, cfg):
        self.cfg = cfg
        
    def validate(self, model, data_loader, device, save_dir) -> dict:
        return self.custom_eval(model, data_loader, device, "val", save_dir)

    def test(self, model, data_loader, device, save_dir) -> dict:
        return self.custom_eval(model, data_loader, device, "test", save_dir)
    
    def custom_eval(self, model, data_loader, device, prefix, save_dir) -> dict:
        metrics = self._run(model, data_loader, device, save_dir)
        metric_dict = {}
        for key, value in metrics.items():
            metric_dict[f"{prefix}/{key}"] = value
        return metric_dict
    
    @torch.no_grad()
    def _run(self, model, data_loader, device, save_dir=None):
        model.eval()
        # part = ['sc', 'gm']
        model.to(device=device)
        num_val_batches = len(data_loader)

        dice_metric = DiceMetric(include_background=True, reduction="none")
        jc_metric = MeanIoU(include_background=True, reduction="mean")
        hd_metric = HausdorffDistanceMetric(include_background=True, reduction="none", percentile=95)
        asd_metric = SurfaceDistanceMetric(include_background=True, reduction="none")

        # iterate over the validation set
        for image_paths, images, masks in tqdm(data_loader, total=num_val_batches, desc='Evaluation', unit='batch', leave=False):
            # move images and labels to correct device and type
            images = images.to(device=device)
            n_imgs = images.size(0)
            masks = masks.unsqueeze(1).float().to(device=device)

            output = model(images)
            pred_label = torch.sigmoid(output).ge(0.5)

            # # Loop over the number of classes (cup and disc in this case)
            # reference from the code of the paper: 
            # DoFE: Domain-oriented Feature Embedding for Generalizable Fundus Image Segmentation on Unseen Datasets
            for i in range(pred_label.shape[1]):  
                if pred_label[:, i].float().sum() < 1e-4:
                    # If no significant foreground is detected, you might want to set pred_label to all zeros
                    pred_label[:, i] = torch.zeros_like(pred_label[:, i])
            
            dice_values = dice_metric(y_pred=pred_label, y=masks)
            jc_metric(pred_label, masks)
            hd_metric(y_pred=pred_label, y=masks)
            asd_metric(y_pred=pred_label, y=masks)
            # save the predicted results
            if save_dir is not None:
                for i in range(n_imgs):
                    img_path = image_paths[i]
                    parent_folder = Path(img_path).parent
                    img_name = Path(img_path).name.split(".")[0]
                    dice_value = dice_values[i].cpu().numpy()
                    saved_path = parent_folder / "result" / save_dir / f"{img_name}_dice_{dice_value.mean():.4f}.png"
                    Path(saved_path).parent.mkdir(parents=True, exist_ok=True)
                    img = Image.open(img_path)
                    img = np.array(img)
                    draw_mask_and_save(img, pred_label[i], saved_path)
                    
        dc = dice_metric.aggregate().mean(axis=0)
        bc_dc = dc[0].item()
        avg_dc = dc.mean().item()

        jc = jc_metric.aggregate().item()

        hd_aggregated = hd_metric.aggregate()
        asd_aggregated = asd_metric.aggregate()

        zero_value_replace = 100
        for i in range(hd_aggregated.shape[1]):
            # mask of valid values
            hd_aggregated[:,i] = torch.where(torch.isinf(hd_aggregated[:,i]) | torch.isnan(hd_aggregated[:,i]), zero_value_replace, hd_aggregated[:,i])
            asd_aggregated[:,i] = torch.where(torch.isinf(asd_aggregated[:,i]) | torch.isnan(asd_aggregated[:,i]), zero_value_replace, asd_aggregated[:,i])

        hd = hd_aggregated.mean().item()
        asd = asd_aggregated.mean().item()

        metrics = {
            "tumor": bc_dc,
            "avg_dc": avg_dc,
            "jc": jc,
            "hd": hd,
            "asd": asd,
        }
        dice_metric.reset()
        jc_metric.reset()
        hd_metric.reset()
        asd_metric.reset()
        return metrics
    
    @torch.no_grad()
    def cal_kl_loss(self, local_model, global_model, data_loader, device):
        local_model.eval()
        local_model.to(device=device)
        global_model.eval()
        global_model.to(device=device)

        total_loss = 0.0
        total_elements = 0
        epsilon = 1e-8

        for _, images in tqdm(data_loader, desc='KL Calculation', unit='batch', leave=False):
            images = images.to(device=device)
            output_local = local_model(images)  # Shape: [batch_size, num_classes, H, W]
            output_global = global_model(images)

            probs_local = torch.sigmoid(output_local).clamp(min=epsilon, max=1 - epsilon)
            probs_global = torch.sigmoid(output_global).clamp(min=epsilon, max=1 - epsilon)

            log_probs_local = torch.log(probs_local)

            kl_loss = F.kl_div(
                log_probs_local,    
                probs_global,       
                reduction='sum'     
            )

            total_loss += kl_loss.item()

            batch_size = images.size(0)
            num_classes = output_local.size(1)
            H = output_local.size(2)
            W = output_local.size(3)
            num_elements = batch_size * num_classes * H * W
            total_elements += num_elements

        average_loss = total_loss / total_elements
        # larger average_loss means more difference between local and global model, which means worse generalization gap
        return average_loss