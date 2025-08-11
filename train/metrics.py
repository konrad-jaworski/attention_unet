import torch

class evaluation:
    def __init__(self):
        super().__init__()

    def compute_metrics(self, preds, targets, threshold=0.5, eps=1e-7):
        preds = torch.sigmoid(preds)
        preds = (preds > threshold).float()
        targets = targets.float()

        # Flatten everything except batch
        preds = preds.view(preds.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (preds * targets).sum(dim=1)
        union = (preds + targets).sum(dim=1)
        preds_sum = preds.sum(dim=1)
        targets_sum = targets.sum(dim=1)

        dice = (2 * intersection + eps) / (union + eps)
        precision = (intersection + eps) / (preds_sum + eps)
        recall = (intersection + eps) / (targets_sum + eps)
        f1 = (2 * precision * recall + eps) / (precision + recall + eps)
        iou = (intersection + eps) / (preds_sum + targets_sum - intersection + eps)

        return {
            'dice': dice.mean().item(),
            'precision': precision.mean().item(),
            'recall': recall.mean().item(),
            'f1': f1.mean().item(),
            'iou': iou.mean().item()
        }

    def evaluate_model(self, model, dataloader, treshold, device, model_unet=False):
        model.eval()
        all_metrics = {
            'dice': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'iou': []
        }

        with torch.no_grad():
            for images, masks in dataloader:
                images = images.to(device)
                masks = masks.to(device)
                if model_unet:
                    outputs, _, _, _ = model(images)
                else:
                    outputs = model(images)

                metrics = self.compute_metrics(outputs, masks, treshold)
                for k in all_metrics:
                    all_metrics[k].append(metrics[k])

        averaged = {k: sum(v) / len(v) for k, v in all_metrics.items()}
        return averaged
