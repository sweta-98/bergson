import torch, sys
sys.path.insert(0, '.')
from bergson.gradients import GradientProcessor
from bergson.utils.math import compute_lambda, damped_eigh

run = 'runs/olmo_wmdp'

# Load raw preconditioners (not eigen)
q_proc = GradientProcessor.load(f'{run}/query_preconditioner')
i_proc = GradientProcessor.load(f'{run}/value_preconditioner')

# --- Old λ: using the saved (undamped) eigendecomposition ---
old_lambda = compute_lambda(q_proc.preconditioners_eigen, i_proc.preconditioners_eigen,
target_components=1000)
print(f'Old lambda (undamped eigen): {old_lambda:.6f}')

# Count negatives in old eigen
q_negs = sum((ev[0] < 0).sum().item() for ev in q_proc.preconditioners_eigen.values())
i_negs = sum((ev[0] < 0).sum().item() for ev in i_proc.preconditioners_eigen.values())
q_total = sum(ev[0].numel() for ev in q_proc.preconditioners_eigen.values())
i_total = sum(ev[0].numel() for ev in i_proc.preconditioners_eigen.values())
print(f'Query negative eigvals: {q_negs}/{q_total} ({100*q_negs/q_total:.1f}%)')
print(f'Value negative eigvals: {i_negs}/{i_total} ({100*i_negs/i_total:.1f}%)')

# --- New λ: recompute eigendecompositions with damping ---
print('\nRecomputing eigendecompositions with damping...')
q_eigen_damped = {}
i_eigen_damped = {}

for name in q_proc.preconditioners:
    eigvals, eigvecs = damped_eigh(q_proc.preconditioners[name])
    q_eigen_damped[name] = (eigvals.cpu(), eigvecs.cpu())

for name in i_proc.preconditioners:
    eigvals, eigvecs = damped_eigh(i_proc.preconditioners[name])
    i_eigen_damped[name] = (eigvals.cpu(), eigvecs.cpu())

new_lambda = compute_lambda(q_eigen_damped, i_eigen_damped, target_components=1000)
print(f'New lambda (damped eigen): {new_lambda:.6f}')

# Check negatives in damped
q_negs_d = sum((ev[0] < 0).sum().item() for ev in q_eigen_damped.values())
i_negs_d = sum((ev[0] < 0).sum().item() for ev in i_eigen_damped.values())
print(f'Query negative eigvals (damped): {q_negs_d}')
print(f'Value negative eigvals (damped): {i_negs_d}')

# Show the k-th eigenvalues that determine lambda
import numpy as np
q_all_old = torch.cat([ev[0].float().clamp(min=0) for ev in
q_proc.preconditioners_eigen.values()])
i_all_old = torch.cat([ev[0].float().clamp(min=0) for ev in
i_proc.preconditioners_eigen.values()])
q_all_new = torch.cat([ev[0].float().clamp(min=0) for ev in q_eigen_damped.values()])
i_all_new = torch.cat([ev[0].float().clamp(min=0) for ev in i_eigen_damped.values()])

k = 999
q_sorted_old = torch.sort(q_all_old, descending=True).values
i_sorted_old = torch.sort(i_all_old, descending=True).values
q_sorted_new = torch.sort(q_all_new, descending=True).values
i_sorted_new = torch.sort(i_all_new, descending=True).values

print(f'\nAt k={k+1}:')
print(f'  Old: sigma_query={q_sorted_old[k]:.6f}  sigma_value={i_sorted_old[k]:.6f}')
print(f'  New: sigma_query={q_sorted_new[k]:.6f}  sigma_value={i_sorted_new[k]:.6f}')
print(f'  Old lambda = {i_sorted_old[k]/(q_sorted_old[k]+i_sorted_old[k]):.6f}')
print(f'  New lambda = {i_sorted_new[k]/(q_sorted_new[k]+i_sorted_new[k]):.6f}')