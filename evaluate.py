import torch
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

def compute_ece(y_true, y_prob, n_bins=10):
    if y_true.ndim == 1:
        num_classes = y_prob.shape[1]
        y_true_oh = np.eye(num_classes)[y_true]
    else:
        y_true_oh = y_true
        num_classes = y_true.shape[1]

    eces = []
    for c in range(num_classes):
        bins = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bins[:-1]
        bin_uppers = bins[1:]
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            mask = (y_prob[:, c] > bin_lower) & (y_prob[:, c] <= bin_upper)
            if np.any(mask):
                acc = (y_true_oh[:, c][mask] == (y_prob[:, c][mask] > 0.5)).mean()
                conf = y_prob[:, c][mask].mean()
                ece += np.abs(conf - acc) * mask.mean()
        eces.append(ece)
    return np.mean(eces)

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
        auc = roc_auc_score(labels_np, probs_np, multi_class='ovr')

    ece = compute_ece(labels_np, probs_np)

    return {
        'acc': acc,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'auc': auc,
        'ece': ece
    }
