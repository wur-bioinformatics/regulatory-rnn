import torch
import graphviz as gv
from IPython.display import display


class RegulatoryNetwork:
    def __init__(self, links: torch.Tensor, node_names: list[str]):
        self.links = links
        self.node_names = node_names

    @property
    def n_nodes(self):
        return self.links.shape[1]

    @property
    def n_targets(self):
        return self.links.shape[0]

    def draw(self):
        g = gv.Graph("G", filename="simple_graph.gv", engine="sfdp")

        for source in range(self.n_nodes):
            g.node(str(source), label=self.node_names[source], shape="circle")
            for target in range(self.n_targets):
                weight = self.links[target, source].item()
                if abs(weight) < 0.0:
                    continue
                g.edge(
                    str(source),
                    str(target),
                    weight=str(weight),
                    dir="forward",
                    arrowhead="normal" if weight > 0 else "tee",
                    color="darkgrey",
                    penwidth=str(abs(weight) * 5),
                )

        display(g)
