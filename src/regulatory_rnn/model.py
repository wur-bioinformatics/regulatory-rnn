import torch
import torch.nn as nn
from tqdm.auto import tqdm, trange

from .network import RegulatoryNetwork


class MultiOutputLinear(nn.Module):
    """Linear layer with independent weights for each output, i.e. one 'network' per output"""

    def __init__(self, input_size: int, output_size: int, n_networks: int = 1):
        super().__init__()
        self.weights = nn.Parameter(torch.empty(n_networks, output_size, input_size))
        self.biases = nn.Parameter(torch.empty(n_networks, 1, output_size))
        self._init_params()

    def __repr__(self):
        n_networks, output_dim, input_dim = self.weights.shape
        return f"MultiOutputLinear({n_networks=}, {input_dim=}, {output_dim=})"

    def _init_params(self):
        nn.init.uniform_(self.weights, -1, 1)
        nn.init.uniform_(self.biases, -1, 1)

    def forward(self, x: torch.Tensor):
        """Treat both n_networks and n_cases as 'batch dimension' to matrix multiply everything in one go"""
        n_networks, n_outputs, _ = self.weights.shape
        pred = (x @ self.weights.transpose(-2, -1)) + self.biases
        if n_networks == 1:  # correctly handle shapes for final layer of network
            pred = pred.squeeze(0)
        if n_outputs == 1:
            pred = pred.squeeze(-1).T
        return pred


class LinkMask(nn.Module):
    """Helper class that masks relationships between specific inputs and outputs, taking into account tensor shapes (e.g. when used for batching)"""

    def __init__(
        self,
        n_inputs: int = None,
        n_outputs: int = None,
        link_mask: torch.Tensor = None,
    ):
        super().__init__()
        assert (link_mask is not None) or (
            n_inputs is not None and n_outputs is not None
        ), "Must specify mask shape (n_inputs and n_outputs), or provide mask tensor"
        if link_mask is None:
            link_mask = torch.ones((n_outputs, n_inputs))
        self.link_mask = link_mask

    def __str__(self):
        n_outputs, n_inputs = self.link_mask.shape
        return f"LinkMask({n_inputs=}, {n_outputs=})"

    def forward(self, stimuli: torch.Tensor):
        n_networks, n_cases, n_stimuli = stimuli.shape  # n_networks == n_outputs
        broadcast_mask = self.link_mask.unsqueeze(1).tile(
            1, n_cases, 1
        )  # broadcast to shape [n_networks, n_cases, n_inputs]
        assert broadcast_mask.shape == stimuli.shape, (
            f"{broadcast_mask.shape} != {stimuli.shape=}"
        )
        return broadcast_mask * stimuli


class RegulatoryRNN(nn.Module):
    def __init__(
        self,
        n_targets: int,
        n_stimuli: int,
        n_observed_targets: int,
        hidden_size: int = 16,
        n_networks: int = None,
        stepsize: float = 0.2,
        observed_init_value: float = 0.4,
    ):
        super().__init__()
        assert n_observed_targets <= n_targets, (
            f"{n_observed_targets=} should be smaller than or equal to {n_targets=}"
        )
        if n_networks is None:
            n_networks = n_targets
        assert n_networks == n_targets or n_networks == 1, (
            f"{n_networks=} should be equal to {n_targets=} or 1"
        )

        self.mlp = nn.Sequential(
            MultiOutputLinear(
                n_networks=n_networks,
                input_size=n_targets + n_stimuli,
                output_size=hidden_size,
            ),
            nn.ReLU(),
            MultiOutputLinear(
                n_networks=n_networks, input_size=hidden_size, output_size=hidden_size
            ),
            nn.ReLU(),
            MultiOutputLinear(
                n_networks=n_networks,
                input_size=hidden_size,
                output_size=1 if n_networks == n_targets else n_targets,
            ),
            nn.Sigmoid(),
        )

        self.n_observed_targets = n_observed_targets
        self.hidden_init = nn.Parameter(
            torch.empty((n_targets - n_observed_targets,))
        )  # make this a nn.Parameter to optimize
        self.observed_init = torch.tensor([observed_init_value] * n_observed_targets)
        self.n_targets = n_targets
        self.n_stimuli = n_stimuli
        self.n_networks = n_networks
        self.stepsize = stepsize
        self.training_loss_ = []
        self.learning_rate_ = []
        self._init_params()

    def _init_params(self):
        nn.init.uniform_(self.hidden_init, 0, 1)
        for layer in self.mlp:
            if hasattr(layer, "_init_params"):
                layer._init_params()

    def forward(
        self, stimuli: torch.Tensor, link_mask: torch.Tensor = None
    ) -> torch.Tensor:
        n_stimuli, n_timepoints, _ = stimuli.shape
        if link_mask is None:
            link_mask = LinkMask(
                n_inputs=self.n_stimuli + self.n_targets, n_outputs=self.n_targets
            )
        else:
            link_mask = LinkMask(link_mask=link_mask)
        g = torch.concat(
            [self.observed_init, self.hidden_init]
        ).tile(
            [n_stimuli, 1]
        )  # tiling duplicates to account for batches. Observed always before hidden! (important for plotting later)
        g_pred = torch.zeros(
            [n_stimuli, n_timepoints + 1, g.size(1)]
        )  # initialize empty prediction tensor
        g_pred[:, 0, :] = g  # first prediction elements are initialization values
        for t in range(n_timepoints):
            h = torch.concat(
                [g, stimuli[:, t, :]], dim=1
            ).tile(
                [self.n_networks, 1, 1]
            )  # concat external stimulus and network node values. [n_networks, n_cases, n_nodes]
            masked_h = link_mask(h)  # apply link mask
            f = self.mlp(masked_h)  # calculate production
            g = (
                (self.stepsize * f) + (1 - self.stepsize) * g
            )  # balance production and degradation. dg/dt = f(.) - g --> g_{t+1} = scale * f_{t}(.) + (1 - scale) * g_{t}
            g_pred[:, t + 1, :] = g  # store predicted value
        return g_pred

    def fit(
        self,
        dataset,
        optimizer=None,
        n_steps: int = 200,
        link_mask: torch.Tensor = None,
        reset_before_fit: bool = True,
    ):
        if optimizer is None:
            optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=0.2, total_steps=n_steps
            )
        if reset_before_fit:
            self._init_params()
        pbar = tqdm(total=n_steps)
        for i, (stimuli, targets) in enumerate(dataset):
            pbar.update()
            preds = self(stimuli, link_mask)
            loss = dataset.loss_function(preds, targets)
            self.training_loss_.append(loss.item())
            self.learning_rate_.append(scheduler.get_last_lr()[0])
            self.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            if i == n_steps - 1:
                pbar.close()
                return loss.item()

    def link_mutation_test(self, dataset, link_mask: LinkMask, target_index: int):
        test_stimuli = dataset.test_stimuli()
        plot_pred_wt = self(test_stimuli, link_mask=None)  # .detach()
        plot_pred_ko = self(test_stimuli, link_mask=link_mask)  # .detach()
        effect = torch.mean(
            plot_pred_wt[:, :, target_index] - plot_pred_ko[:, :, target_index]
        )
        return effect

    def infer_network(self, dataset, refit: bool = True, steps_per_iter: int = 400):
        links = torch.zeros((dataset.n_targets, dataset.n_nodes))
        unseen_mask = (links == 0) * 1
        for _ in trange(links.numel()):
            if refit:
                self.fit(dataset, link_mask=unseen_mask, n_steps=steps_per_iter)
            test_links = torch.zeros((dataset.n_targets, dataset.n_nodes))
            for source in range(dataset.n_nodes):
                for target in range(dataset.n_targets):
                    if links[target, source].abs() > 0:
                        continue
                    link_mask = unseen_mask.clone()
                    link_mask[target, source] = 0.0
                    effect = self.link_mutation_test(dataset, link_mask, target)
                    test_links[target, source] = effect
            biggest_effect_indices = (
                test_links.abs() == test_links.abs().max()
            ).nonzero()[0]
            links[*biggest_effect_indices] = test_links[*biggest_effect_indices]
            unseen_mask = (links == 0) * 1
        return RegulatoryNetwork(links, dataset.node_names)
