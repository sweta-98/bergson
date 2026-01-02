

import torch
from torch.optim import Muon, AdamW

class MuonAdamW(torch.optim.Optimizer):
    """
    Hybrid optimizer that applies torch.optim.Muon to 2D hidden matrices
    and torch.optim.AdamW to everything else (embeddings, biases, norms).
    """
    def __init__(self, params, muon_lr=0.02, adam_lr=3e-4, 
                 muon_momentum=0.95, adam_betas=(0.9, 0.95), 
                 adam_eps=1e-8, weight_decay=0.01):
        
        # 1. Split parameters into Muon (2D matrices) and AdamW (others) groups
        muon_params = []
        adam_params = []
        
        for p in params:
            if not p.requires_grad:
                continue
            
            # Exclude bias and huge embeddings/heads
            if p.ndim >= 2 and p.size(0) < 10000: 
                muon_params.append(p)
            else:
                adam_params.append(p)

        # 2. Initialize internal optimizers
        self.optimizers = []
        
        if muon_params:
            self.muon = Muon(
                muon_params, 
                lr=muon_lr, 
                momentum=muon_momentum,
                weight_decay=weight_decay,
                adjust_lr_fn="match_rms_adamw",
            )
            self.optimizers.append(self.muon)
        
        if adam_params:
            self.adam = AdamW(
                adam_params, 
                lr=adam_lr, 
                betas=adam_betas, 
                eps=adam_eps, 
                weight_decay=weight_decay
            )
            self.optimizers.append(self.adam)

        # 3. Combine param_groups so HF Scheduler can see/update all LRs
        self.param_groups = []
        for opt in self.optimizers:
            self.param_groups.extend(opt.param_groups)

        # Initialize base class (dummy) to satisfy type checks
        super().__init__(self.param_groups, {})

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
            
        for opt in self.optimizers:
            opt.step()
            
        return loss

    def zero_grad(self, set_to_none=True):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {
            'muon': self.muon.state_dict() if hasattr(self, 'muon') else None,
            'adam': self.adam.state_dict() if hasattr(self, 'adam') else None
        }

    def load_state_dict(self, state_dict):
        if hasattr(self, 'muon') and state_dict['muon']:
            self.muon.load_state_dict(state_dict['muon'])
        if hasattr(self, 'adam') and state_dict['adam']:
            self.adam.load_state_dict(state_dict['adam'])