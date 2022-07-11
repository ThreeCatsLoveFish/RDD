import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


@torch.no_grad()
def compute_metrics(model_outputs, labels, auc=False):
    """
    Compute the accuracy metrics.
    """
    probs = F.softmax(model_outputs, dim=1)
    acc = (probs.argmax(1) == labels).float().mean()
    auc = roc_auc_score(labels.cpu(), probs[:, 1].cpu()) if auc else 0
    return acc.item(), auc
