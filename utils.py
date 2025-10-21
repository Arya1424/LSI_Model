"""
Utility Functions for Enhanced LeSICiN
"""
import torch
import numpy as np
import json


def compute_metrics(preds, labels):
    """
    Compute evaluation metrics
    Args:
        preds: predictions (batch_size, num_labels) - binary {0, 1}
        labels: ground truth (batch_size, num_labels) - binary {0, 1}
    Returns:
        dict with macro/micro precision, recall, F1, and Jaccard
    """
    preds = preds.numpy() if isinstance(preds, torch.Tensor) else preds
    labels = labels.numpy() if isinstance(labels, torch.Tensor) else labels
    
    num_labels = labels.shape[1]
    
    # Macro metrics (per-label, then average)
    macro_p = []
    macro_r = []
    macro_f1 = []
    
    for i in range(num_labels):
        label_preds = preds[:, i]
        label_true = labels[:, i]
        
        tp = np.sum((label_preds == 1) & (label_true == 1))
        fp = np.sum((label_preds == 1) & (label_true == 0))
        fn = np.sum((label_preds == 0) & (label_true == 1))
        
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        
        macro_p.append(p)
        macro_r.append(r)
        macro_f1.append(f1)
    
    # Micro metrics (aggregate all, then compute)
    tp_total = np.sum((preds == 1) & (labels == 1))
    fp_total = np.sum((preds == 1) & (labels == 0))
    fn_total = np.sum((preds == 0) & (labels == 1))
    
    micro_p = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
    micro_r = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0
    
    # Jaccard Index (IoU)
    intersection = np.sum((preds == 1) & (labels == 1), axis=1)
    union = np.sum((preds == 1) | (labels == 1), axis=1)
    jaccard = np.mean(intersection / (union + 1e-10))
    
    return {
        'macro': {
            'precision': np.mean(macro_p) * 100,
            'recall': np.mean(macro_r) * 100,
            'f1': np.mean(macro_f1) * 100
        },
        'micro': {
            'precision': micro_p * 100,
            'recall': micro_r * 100,
            'f1': micro_f1 * 100
        },
        'jaccard': jaccard * 100
    }


def save_checkpoint(state, filepath):
    """Save model checkpoint"""
    torch.save(state, filepath)
    print(f"✓ Checkpoint saved to {filepath}")


def load_checkpoint(filepath, model=None, optimizer=None):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location='cpu')
    
    if model is not None and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


def print_metrics(metrics, prefix=""):
    """Pretty print metrics"""
    print(f"{prefix}Macro Metrics:")
    print(f"  Precision: {metrics['macro']['precision']:.2f}%")
    print(f"  Recall:    {metrics['macro']['recall']:.2f}%")
    print(f"  F1:        {metrics['macro']['f1']:.2f}%")
    
    print(f"\n{prefix}Micro Metrics:")
    print(f"  Precision: {metrics['micro']['precision']:.2f}%")
    print(f"  Recall:    {metrics['micro']['recall']:.2f}%")
    print(f"  F1:        {metrics['micro']['f1']:.2f}%")
    
    print(f"\n{prefix}Jaccard Index: {metrics['jaccard']:.2f}%")


def save_predictions(predictions, filepath):
    """Save predictions to JSONL file"""
    import jsonlines
    
    with jsonlines.open(filepath, mode='w') as writer:
        for pred in predictions:
            writer.write(pred)
    
    print(f"✓ Predictions saved to {filepath}")


def load_predictions(filepath):
    """Load predictions from JSONL file"""
    import jsonlines
    
    predictions = []
    with jsonlines.open(filepath) as reader:
        for obj in reader:
            predictions.append(obj)
    
    return predictions


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_gpu_memory():
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        return {
            'allocated': torch.cuda.memory_allocated() / 1e9,
            'reserved': torch.cuda.memory_reserved() / 1e9,
            'max_allocated': torch.cuda.max_memory_allocated() / 1e9
        }
    return None


def print_gpu_memory():
    """Print GPU memory usage"""
    mem = get_gpu_memory()
    if mem:
        print(f"GPU Memory:")
        print(f"  Allocated: {mem['allocated']:.2f} GB")
        print(f"  Reserved:  {mem['reserved']:.2f} GB")
        print(f"  Max Used:  {mem['max_allocated']:.2f} GB")


if __name__ == '__main__':
    # Test metrics computation
    print("Testing utility functions...")
    
    # Create dummy data
    preds = torch.randint(0, 2, (100, 50)).float()
    labels = torch.randint(0, 2, (100, 50)).float()
    
    # Compute metrics
    metrics = compute_metrics(preds, labels)
    
    print("\nMetrics Test:")
    print_metrics(metrics)
    
    print("\n✓ Utils test passed!")