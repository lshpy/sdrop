import torch
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

def compute_ece(y_true, y_prob, n_bins=15):
    """
    Expected Calibration Error.

    For single-label multi-class classification, it is computed on **top-1 confidence** as usual:
    the maximum predicted probability is the confidence, and |accuracy - mean confidence|
    per confidence bin is summed, weighted by the fraction of samples in the bin.

        ECE = sum_b (|B_b| / N) * | acc(B_b) - conf(B_b) |

    Only for multi-label input (y_true a 2-D indicator matrix) is it computed one-vs-rest per class,
    with the bin accuracy given by the **empirical positive frequency**.

    [Bug in an earlier implementation] In the one-vs-rest bins, acc was computed as
    `y_true == (p > 0.5)`, i.e. agreement with a 0.5 threshold. With many classes (CIFAR-100) most
    samples fall into the lowest bin, where acc ~ 1 - 1/K and conf ~ 0, so ECE was
    fixed at (1 - 1/K) regardless of the model (0.98 for every run).
    """
    y_prob = np.asarray(y_prob)
    y_true = np.asarray(y_true)

    if y_true.ndim == 1:
        # ---- single-label multi-class: top-1 confidence ----
        conf = y_prob.max(axis=1)
        pred = y_prob.argmax(axis=1)
        correct = (pred == y_true).astype(float)
        return _binned_ece(conf, correct, n_bins)

    # ---- multi-label: one-vs-rest per class, acc = empirical positive frequency ----
    eces = []
    for c in range(y_true.shape[1]):
        eces.append(_binned_ece(y_prob[:, c], y_true[:, c].astype(float), n_bins))
    return float(np.mean(eces))


def _binned_ece(conf, correct, n_bins=15):
    """Bin conf (confidence) and correct (0/1) and compute ECE."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(conf)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf > lo) & (conf <= hi)
        if not np.any(mask):
            continue
        ece += np.abs(correct[mask].mean() - conf[mask].mean()) * (mask.sum() / n)
    return float(ece)


def evaluate(model, test_loader, device='cuda'):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            is_multilabel = target.ndim > 1 and target.size(1) > 1

            if is_multilabel:
                probs = torch.sigmoid(output)
                preds = (probs > 0.5).float()
            else:
                probs = torch.softmax(output, dim=1)
                preds = torch.argmax(probs, dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(target.cpu())
            all_probs.append(probs.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_probs = torch.cat(all_probs)

    labels_np = all_labels.numpy()
    preds_np = all_preds.numpy()
    probs_np = all_probs.numpy()

    is_multilabel = len(labels_np.shape) == 2 and labels_np.shape[1] > 1

    if is_multilabel:
        acc = (all_preds == all_labels).all(dim=1).float().mean().item() * 100
        f1_macro = f1_score(labels_np, preds_np, average='macro')
        f1_micro = f1_score(labels_np, preds_np, average='micro')
        auc = roc_auc_score(labels_np, probs_np)
    else:
        acc = accuracy_score(labels_np, preds_np) * 100
        f1_macro = f1_score(labels_np, preds_np, average='macro')
        f1_micro = f1_score(labels_np, preds_np, average='micro')
        # Binary classification (waterbirds) must pass only the positive-class column; sklearn fails on (N,2).
        if probs_np.ndim == 2 and probs_np.shape[1] == 2:
            auc = roc_auc_score(labels_np, probs_np[:, 1])
        else:
            auc = roc_auc_score(labels_np, probs_np, multi_class='ovr')

    ece = compute_ece(labels_np, probs_np)

    return {
        'acc': acc,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'auc': auc,
        'ece': ece
    }
