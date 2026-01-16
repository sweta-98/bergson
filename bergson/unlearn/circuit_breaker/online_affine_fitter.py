import torch
import torch.nn as nn
from tqdm import tqdm

class OnlineAffineFitter:
    """
    Accumulates statistics to solve Y = XW + b using Ridge Regression
    without holding all activations in memory.
    """
    def __init__(self, hidden_dim, device="cpu", alpha=0.01):
        self.dim = hidden_dim
        self.device = device
        self.alpha = alpha
        
        # Accumulators (using float64 for precision during summation)
        self.n_samples = 0
        self.sum_x = torch.zeros(hidden_dim, dtype=torch.float64, device=device)
        self.sum_y = torch.zeros(hidden_dim, dtype=torch.float64, device=device)
        self.xtx = torch.zeros((hidden_dim, hidden_dim), dtype=torch.float64, device=device)
        self.xty = torch.zeros((hidden_dim, hidden_dim), dtype=torch.float64, device=device)

    def update(self, x, y):
        """
        x: [Batch * Seq, Dim] - Source Activations
        y: [Batch * Seq, Dim] - Target Activations
        """
        x = x.to(self.device).double()
        y = y.to(self.device).double()
        
        self.n_samples += x.shape[0]
        self.sum_x += x.sum(dim=0)
        self.sum_y += y.sum(dim=0)
        
        # X^T X
        self.xtx += x.T @ x
        # X^T Y
        self.xty += x.T @ y

    def solve(self):
        """
        Solves for W and b.
        Returns a torch.nn.Linear module initialized with the mapping.
        """
        # Compute Means
        mu_x = (self.sum_x / self.n_samples).float()
        mu_y = (self.sum_y / self.n_samples).float()
        
        # Compute Centered Covariances
        # Cov(X,X) = sum(x^2) - n * mu_x^2
        # We need to be careful with shapes here for outer product
        mu_x_64 = self.sum_x / self.n_samples
        mu_y_64 = self.sum_y / self.n_samples
        
        cov_xx = self.xtx - self.n_samples * torch.outer(mu_x_64, mu_x_64)
        cov_xy = self.xty - self.n_samples * torch.outer(mu_x_64, mu_y_64)
        
        # Ridge Regression: W = (Cov_XX + alpha*I)^-1 @ Cov_XY
        eye = torch.eye(self.dim, device=self.device, dtype=torch.float64)
        cov_xx_reg = cov_xx + (self.alpha * eye)
        
        # Solve
        W = torch.linalg.solve(cov_xx_reg, cov_xy).float()
        
        # Calculate Bias: b = mu_y - mu_x @ W
        b = mu_y - (mu_x @ W)
        
        # Create Linear Module
        module = nn.Linear(self.dim, self.dim, bias=True)
        with torch.no_grad():
            module.weight.copy_(W.T) # nn.Linear stores as (Out, In)
            module.bias.copy_(b)
        
        return module.to(mu_x.device)

def train_affine_transform(
    source_model,
    target_model,
    tokenizer,
    dataset,
    target_layers,
    num_examples=100_000,
    batch_size=4,
    device="cuda",
    alpha=0.01
):
    """
    Passes data through both models and trains affine transforms for specified layers.
    Returns: Dict[layer_idx, nn.Linear]
    """
    print(f"🔄 Training Affine Transform (Old -> New) on {num_examples} examples...")
    
    # 1. Setup Fitters for each layer
    hidden_dim = target_model.config.hidden_size
    fitters = {
        layer: OnlineAffineFitter(hidden_dim, device=device, alpha=alpha) 
        for layer in target_layers
    }
    
    # 2. Prepare Data
    # Assuming dataset supports slicing or we just take the first N
    if hasattr(dataset, "select"):
        subset = dataset.select(range(min(num_examples, len(dataset))))
    else:
        subset = [dataset[i] for i in range(min(num_examples, len(dataset)))]

    def collate(batch_items, device):
        # input_ids_circuit_breaker, attention_mask_circuit_breaker
        
        # 1. Process Input IDs: Ensure they are 1D before stacking
        input_ids_list = []
        for x in batch_items:
            tensor = torch.tensor(x['input_ids'], dtype=torch.long)
            # Remove extra dimensions (e.g., if shape is [1, 1024] -> [1024])
            if tensor.ndim > 1:
                tensor = tensor.view(-1)
            input_ids_list.append(tensor)
            
        input_ids = torch.stack(input_ids_list).to(device)

        # 2. Process Mask: Ensure 1D and use Bool dtype (from previous fix)
        mask_list = []
        for x in batch_items:
            tensor = torch.tensor(x['attention_mask'])
            if tensor.ndim > 1:
                tensor = tensor.view(-1)
            mask_list.append(tensor)

        attention_mask = torch.stack(mask_list).to(device=device, dtype=torch.bool)

        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    # 3. Loop
    source_model.eval()
    target_model.eval()
    
    num_batches = (len(subset) + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Collecting Activations"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(subset))
            batch = subset[start_idx:end_idx]
            
            inputs = collate(batch, device)

            mask = inputs["attention_mask"].bool().reshape(-1) # [Batch*Seq]
            
            # Forward Source (Old)
            out_source = source_model(**inputs, output_hidden_states=True)
            
            # Forward Target (New)
            out_target = target_model(**inputs, output_hidden_states=True)
            
            # Update Fitters
            for layer_idx in target_layers:
                # [Batch, Seq, Dim]
                act_src = out_source.hidden_states[layer_idx]
                act_tgt = out_target.hidden_states[layer_idx]
                
                # Flatten: [Batch*Seq, Dim]
                flat_src = act_src.flatten(0, 1)
                flat_tgt = act_tgt.flatten(0, 1)
                
                # Filter Padding
                clean_src = flat_src[mask]
                clean_tgt = flat_tgt[mask]
                
                fitters[layer_idx].update(clean_src, clean_tgt)

            # Cleanup
            del out_source, out_target, inputs
            torch.cuda.empty_cache()

    # 4. Solve
    print("📐 Solving Linear Systems...")
    affine_transforms = {}
    for layer_idx, fitter in fitters.items():
        affine_transforms[layer_idx] = fitter.solve()
    
    print("✅ Affine Transformation Trained.")
    return affine_transforms