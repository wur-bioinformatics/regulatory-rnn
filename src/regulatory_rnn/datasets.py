import torch
from torch.utils.data import IterableDataset
import pandas as pd


def adapt_pulse_double_exp(
    time_points: torch.Tensor, height: float = 1.0
) -> torch.tensor:
    # inputs: scalars
    # output: np.array of shape=time_points
    # double exponential response curve
    # used for results in figs. 1&2
    xs0 = torch.linspace(0.0, time_points - 1, time_points)
    y = height * 2 * (torch.exp(-xs0 / 6) - torch.exp(-xs0 / 3))
    return y.unsqueeze(1).unsqueeze(0)


def spikes(
    time_points: torch.Tensor, position: torch.Tensor, height: float, duration: float
):
    # inputs: np.array of position,height,duration for each triangular pulse
    # arbitrary number of pulses
    # pulse positions arranged in ascending order
    # output: np.array of shape=time_points
    # used for results in fig. 3-4

    num_peaks = position.shape[0]
    xs0 = torch.linspace(0.0, (time_points - 1), time_points)
    xs = torch.tile(
        torch.expand_dims(xs0, 0), [num_peaks, 1]
    )  # [num_peaks, time_points]
    y0 = (
        torch.expand_dims((position / duration + 1) * 2 * height, 1)
        - 2 * torch.expand_dims(height / duration, 1) * xs
    )
    y0_ = torch.expand_dims(height, 1) - torch.abs(y0 - torch.expand_dims(height, 1))
    position_next = torch.concatenate(
        [position[1:], time_points + torch.ones(1)], axis=0
    )  # [num_peaks]
    mask = torch.float32(
        (xs >= torch.expand_dims(position, 1))
        * (xs < torch.expand_dims(position_next, 1))
        * (xs < torch.expand_dims(position + duration, 1))
    )
    y1 = y0_ * mask  # [num_peaks, time_points]
    y = torch.float32(torch.sum(y1, axis=0))
    return y


class AdaptPulseDataset(IterableDataset):
    """Dataset that provides three levels of adaptation response"""

    n_stimuli = 1
    n_targets = 2
    n_observed_targets = 1
    stimulus_names = ["I"]
    target_names = ["g1", "g2"]
    n_cases = 3

    def __init__(self, n_timepoints: int):
        super().__init__()
        self.n_timepoints = n_timepoints
        assert len(self.node_names) == self.n_stimuli + self.n_targets

    def __iter__(self):
        return self

    def __next__(self):
        return self.sample()

    @property
    def n_nodes(self):
        return self.n_stimuli + self.n_targets

    @property
    def node_names(self):
        return self.target_names + self.stimulus_names

    def sample(self):
        stimulus_0 = torch.ones([1, self.n_timepoints, self.n_stimuli]) * 0.1
        target_0 = torch.ones([1, self.n_timepoints, 1]) * 0.4

        stimulus_1 = torch.ones([1, self.n_timepoints, self.n_stimuli])
        target_1 = adapt_pulse_double_exp(self.n_timepoints, height=1.0) + 0.4

        random_stimulus_level = torch.rand((1,))
        stimulus_2 = (
            torch.ones([1, self.n_timepoints, self.n_stimuli]) * random_stimulus_level
        )
        target_2 = (
            adapt_pulse_double_exp(self.n_timepoints, height=random_stimulus_level)
            + 0.4
        )

        stimuli = torch.concat([stimulus_0, stimulus_1, stimulus_2])
        targets = torch.concat([target_0, target_1, target_2])
        return stimuli, targets

    def test_stimuli(self, n_cases: int = 4):
        test_stimuli = 0.1 * torch.ones(
            [n_cases, self.n_timepoints + 20, self.n_stimuli]
        )
        test_stimuli[:, 20:, 0] = torch.linspace(1 / n_cases, 1.0, n_cases).reshape(
            [n_cases, 1]
        )
        return test_stimuli

    @staticmethod
    def loss_function(prediction, target):
        stationary_loss = ((prediction[0] - prediction[0, 0]) ** 2).sum()
        reconstruction_loss = ((prediction[:, 1:, 1:] - target) ** 2).sum()
        loss = (stationary_loss + reconstruction_loss).sqrt()
        return loss


class GapGeneDataset(IterableDataset):
    """Dataset that provides Drosophila embryo gap gene data"""

    n_stimuli = 2
    n_targets = 4
    n_observed_targets = 4
    stimulus_names = ["Bcd", "Tor"]
    target_names = ["hb", "kr", "kni", "gt"]
    n_cases = 91

    def __init__(self, n_timepoints: int = 30):
        super().__init__()
        self.n_timepoints = n_timepoints
        assert len(self.node_names) == self.n_stimuli + self.n_targets

        # define fixed stimuli
        x = torch.linspace(0.05, 0.95, self.n_cases)  # spatial range
        bcd_init = torch.exp(-x / 0.2).reshape([1, self.n_cases, 1])
        tsl_init = (torch.exp(-x / 0.05) + torch.exp(-(1 - x) / 0.05)).reshape(
            [1, self.n_cases, 1]
        )
        self.stimuli = (
            torch.concat([bcd_init, tsl_init], dim=-1)
            .reshape([self.n_cases, 1, self.n_stimuli])
            .tile([1, self.n_timepoints, 1])
        )

        # load target gene responses
        df = pd.read_csv(
            "https://raw.githubusercontent.com/sjx93/rnn_for_gene_network_2020/refs/tags/v1.0/fig5_gap_gene_patterning/Data_frame/FlyEX_nc14T7_gap-genes.csv",
            header=None,
        )
        self.targets = torch.tensor(df.values[:, : self.n_targets])

    def __iter__(self):
        return self

    def __next__(self):
        return self.sample()

    @property
    def n_nodes(self):
        return self.n_stimuli + self.n_targets

    @property
    def node_names(self):
        return self.target_names + self.stimulus_names

    def sample(self):
        return self.stimuli, self.targets

    def test_stimuli(self, n_cases: int = 4):
        return self.stimuli

    @staticmethod
    def loss_function(prediction, target):
        loss = ((prediction[:, -1, :] - target) ** 2).sum().sqrt()
        return loss
