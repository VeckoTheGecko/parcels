import numpy as np
from scipy.spatial import KDTree

from parcels.interaction.base_neighbor import BaseFlatNeighborSearch


class ScipyFlatNeighborSearch(BaseFlatNeighborSearch):
    name = "scipy"

    def __init__(self, interaction_distance, interaction_depth,
                 values=None):
        super().__init__(interaction_distance, interaction_depth, values)
        if self._values is not None:
            self._kdtree = KDTree(values.T)

    def find_neighbors_by_coor(self, coor):
        corrected_coor = (coor/self.inter_dist).reshape(-1)
        rel_idx = np.array(self._kdtree.query_ball_point(corrected_coor, r=1))
        neighbor_idx = self.active_idx[rel_idx]
        return neighbor_idx, np.vstack(self._distance(coor, neighbor_idx))

    def rebuild(self, values=None, active_mask=-1):
        super().rebuild(values, active_mask)
        self._corrected_values = values[:, self._active_idx]/self.inter_dist
        self._kdtree = KDTree(self._corrected_values.T)
