import torch

def supervised_step(model, batch, criterion, device):
    """
    Performs a single supervised training step.
    
    Args:
        model: The PTCNet model.
        batch: A tuple (X, y, time_masks, feat_masks) from the DataLoader.
        criterion: The loss function (e.g., BCEWithLogitsLoss).
        device: The device to run on (cpu or cuda).
        
    Returns:
        loss: The computed loss scalar.
        logits: The raw model outputs (B, 1).
        y: The targets (B, 1).
    """
    X, y, time_masks, feat_masks = batch
    
    X = X.to(device)                  # (B, F, T)
    y = y.to(device).float()          # (B, 1)
    feat_masks = feat_masks.to(device) # (B, F)
    
    # time_masks are currently unused by PTCNet (padding is zero),
    # but available if we switch to attention later.
    
    # Forward pass
    # We pass feature_mask to zero-out contributions from padded features
    logits = model(X, feature_mask=feat_masks)  # shape (B,)

    logits = logits.unsqueeze(-1)     # match (B, 1)
    loss = criterion(logits, y)

    return loss, logits.detach(), y.detach()
