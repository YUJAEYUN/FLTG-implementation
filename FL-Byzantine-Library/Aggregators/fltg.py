import torch
import torch.nn.functional as F
from .base import _BaseAggregator
from torch.utils.data import DataLoader


class FLTG(_BaseAggregator):
    """
    FLTG: Byzantine-Robust Federated Learning via Angle-Based Defense and Non-IID-Aware Weighting

    This aggregator implements the FLTG algorithm which combines:
    1. ReLU-clipped cosine similarity filtering using server's root dataset
    2. Dynamic reference selection based on previous global model
    3. Non-IID aware weighting inversely proportional to angle deviation
    4. Magnitude normalization to suppress malicious scaling
    """

    def __init__(self, root_dataset, model, args, device):
        super(FLTG, self).__init__()
        self.loader = DataLoader(root_dataset, batch_size=64, shuffle=True)
        self.model = model
        self.args = args
        self.device = device
        self.server_gradient = torch.zeros(self.count_parameters(), device=self.device)
        self.prev_global_gradient = torch.zeros(self.count_parameters(), device=self.device)

    def count_parameters(self):
        """Count total number of parameters in model"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def local_step(self, batch):
        """Compute gradient on server's root dataset"""
        device = self.device
        x, y = batch
        x, y = x.to(device), y.to(device)
        self.model.zero_grad()
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        self.step_sgd()

    def step_sgd(self):
        """Extract gradients from model into server_gradient buffer"""
        args = self.args
        last_ind = 0
        grad_mult = 1 - args.Lmomentum if args.worker_momentum else 1

        for p in self.model.parameters():
            if p.requires_grad:
                d_p = p.grad
                if args.wd != 0:
                    d_p = d_p.add(p.data, alpha=args.wd)

                length, dims = d_p.numel(), d_p.size()
                buf = self.server_gradient[last_ind:last_ind + length].view(dims).detach()
                buf.mul_(args.Lmomentum)
                buf.add_(torch.clone(d_p).detach(), alpha=grad_mult)
                self.server_gradient[last_ind:last_ind + length] = buf.flatten()
                last_ind += length

    def __call__(self, inputs):
        """
        FLTG aggregation algorithm

        Args:
            inputs: List of client gradient tensors

        Returns:
            Aggregated gradient tensor
        """
        # Step 1: Compute server gradient on root dataset
        for data in self.loader:
            self.local_step(data)
            break

        g0 = self.server_gradient  # Server's gradient (g_0^t)
        g_prev = self.prev_global_gradient  # Previous global gradient (g^{t-1})

        # Step 2: ReLU-clipped cosine similarity filtering
        # Filter out clients whose updates are in opposite direction to server
        cos_sims_with_server = [F.cosine_similarity(g, g0, dim=0) for g in inputs]
        filtered_indices = [i for i, sim in enumerate(cos_sims_with_server) if sim > 0]

        # If all clients filtered out, return server gradient
        if len(filtered_indices) == 0:
            self.prev_global_gradient = g0.clone()
            return g0

        filtered_inputs = [inputs[i] for i in filtered_indices]

        # Step 3: Dynamic reference selection
        # Select client with minimum cosine similarity to previous global gradient
        if torch.norm(g_prev) > 0:
            cos_sims_with_prev = [F.cosine_similarity(g, g_prev, dim=0) for g in filtered_inputs]
            ref_idx = cos_sims_with_prev.index(min(cos_sims_with_prev))
        else:
            # First round: use first filtered client as reference
            ref_idx = 0

        g_ref = filtered_inputs[ref_idx]  # Reference gradient

        # Step 4: Non-IID aware weighting
        # Compute scores as 1 - cos_sim(g_i, g_ref)
        # Higher deviation from reference gets higher weight
        scores = []
        for g in filtered_inputs:
            cos_sim_with_ref = F.cosine_similarity(g, g_ref, dim=0)
            score = 1.0 - cos_sim_with_ref
            scores.append(F.relu(score))  # Ensure non-negative

        # Normalize scores to sum to 1
        total_score = sum(scores)
        if total_score > 0:
            weights = [s / total_score for s in scores]
        else:
            # If all scores are zero, use uniform weights
            weights = [1.0 / len(filtered_inputs) for _ in filtered_inputs]

        # Step 5: Magnitude normalization
        # Normalize all client updates to have same magnitude as server gradient
        g0_norm = torch.norm(g0)
        normalized_inputs = []
        for g in filtered_inputs:
            g_norm = torch.norm(g)
            if g_norm > 0:
                normalized_g = g * (g0_norm / g_norm)
            else:
                normalized_g = g
            normalized_inputs.append(normalized_g)

        # Step 6: Weighted aggregation
        aggregated = sum(w * g for w, g in zip(weights, normalized_inputs))

        # Store current aggregated gradient for next round
        self.prev_global_gradient = aggregated.clone()

        return aggregated