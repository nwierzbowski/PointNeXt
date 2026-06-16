"""Training utilities."""
import torch


def save_checkpoint(model, optimizer, epoch, loss, path, scheduler=None, yaml_content=None,
                    best_metric=None, best_metric_value=None):
    """Save model checkpoint."""
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'yaml_content': yaml_content,
        'best_metric': best_metric,
        'best_metric_value': best_metric_value,
    }
    torch.save(state, path)
