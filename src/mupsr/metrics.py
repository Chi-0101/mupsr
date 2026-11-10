"""Full-catalogue ranking metrics with exact tie handling.

Tied scores receive their expected metric under a uniform random permutation of the tie group,
identical to the reference definition used by the frozen protocol (one positive per query).
"""
import torch


@torch.no_grad()
def ranking_metrics(scores, target_column, ks=(5, 10, 20)):
    """scores (Q, N) with -inf for non-candidates; target_column (Q,). Returns per-query R@k and N@k."""
    pos = scores.gather(1, target_column[:, None])
    g, e = (scores > pos).sum(1).double(), (scores == pos).sum(1).double()
    disc = torch.cat((torch.zeros(1, device=scores.device, dtype=torch.float64),
                      torch.cumsum(1/torch.log2(torch.arange(2, max(ks)+2, device=scores.device,
                                                             dtype=torch.float64)), 0)))
    out = {}
    for k in ks:
        out[f'R@{k}'] = ((k-g).clamp(0)).minimum(e)/e
        out[f'N@{k}'] = (disc[(g+e).clamp(max=k).long()]-disc[g.clamp(max=k).long()])/e
    return out
