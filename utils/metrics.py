import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


@torch.no_grad()
def compute_metrics(model_outputs, labels, auc=False):
    """
    Compute the accuracy metrics.
    """
    real_probs = F.softmax(model_outputs, dim=1)[:, 1]
    bin_preds = (real_probs > 0.5).int()
    bin_labels = (labels != 0).int()
    acc = (bin_preds == bin_labels).float().mean()
    auc = roc_auc_score(bin_labels.cpu(), real_probs.cpu()) if auc else 0

    return acc.item(), auc
