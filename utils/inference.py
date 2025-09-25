import torch

def predict_koopman_trajectory(model, X0, steps=50, device=None):
    """Predict future states using a Koopman operator model with an autoencoder.

    Args:
        model (torch.nn.ModuleDict): A ModuleDict containing 'ae' (autoencoder) and 'knet' (Koopman operator).
        X0 (torch.Tensor or array-like): Initial state data.
        steps (int, optional): Number of prediction steps. Defaults to 50.
        device (torch.device, optional): Device to perform computation on. If None, inferred from model.

    Returns:
        torch.Tensor: Predicted trajectory of shape (steps+1, input_size).
    """
    
    model.eval()
    # Infer device from model if not provided
    if device is None:
        device = next(model.parameters()).device

    # Convert input to tensor if necessary and ensure correct device
    if not isinstance(X0, torch.Tensor):
        X0 = torch.tensor(X0, dtype=torch.float32, device=device)

    # Initialize output tensor
    Xpred = torch.zeros((steps + 1, *X0.shape), device=device)
    Xpred[0, :] = X0

    # Perform prediction
    with torch.no_grad():
        Yencoded = model['ae'].encoder(X0)  # Encode initial state
        for index in range(1, steps + 1):
            Ypred = model['knet'](Yencoded)  # Apply Koopman operator
            Xpred[index, :] = model['ae'].decoder(Ypred)  # Decode to original space
            Yencoded = Ypred  # Update encoded state for next step

    return Xpred


def reconstruct_trajectory(model, X, device=None):
    """Reconstruct a trajectory using an autoencoder model.

    Args:
        model (torch.nn.ModuleDict): A ModuleDict containing 'ae' (autoencoder).
        X (torch.Tensor or array-like): State data.
        device (torch.device, optional): Device to perform computation on. If None, inferred from model.

    Returns:
        torch.Tensor: Reconstructed trajectory of shape (steps, input_size).
    """
    
    model.eval()
    # Infer device from model if not provided
    if device is None:
        device = next(model.parameters()).device

    # Convert input to tensor if necessary and ensure correct device
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32, device=device)

    # Perform reconstruction
    with torch.no_grad():
        Y , Xrecon = model['ae'](X)

    return Xrecon