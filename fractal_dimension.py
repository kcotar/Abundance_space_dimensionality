import numpy as np


class Fractal:
    def __init__(self, data, dist='euclidean', verbose=False):
        self.data = data
        self.data_size = np.shape(data)
        self.dist_type = dist
        self.verbose = verbose
        self.data_distances = None

    def _distances(self, ref_obs, target_obs):
        dist = target_obs - ref_obs
        if self.dist_type is 'euclidean':
            dist = np.sqrt(np.sum(np.power(dist, 2), 1))
        return dist

    def _compute_distances(self):
        # compute distances for all possible pairs of points in array of size (N, D)
        final_dist = np.ndarray((self.data_size[0], self.data_size[0]), dtype=np.float16)
        final_dist.fill(np.nan)
        if self.verbose:
            print ' Calculating distance for row:'
        for i_row in range(1, self.data_size[0]-1):
            if self.verbose:
                if i_row % 5000 == 0:
                    print i_row
            dist_for_row = self._distances(self.data[i_row, :], self.data[i_row+1:, :])
            final_dist[i_row, 0:len(dist_for_row)] = np.float16(dist_for_row)
        self.data_distances = final_dist

    def correlation_dimension(self, eps_limit=None, eps_steps=100):
        # compute distance pairs
        if self.data_distances is None:
            self._compute_distances()
        # proportion of the distances that are less than or equal to epsilon
        if eps_limit is None:
            eps_limit = [np.nanmin(self.data_distances), np.nanmax(self.data_distances)]
        eps_step = (eps_limit[1] - eps_limit[0]) / eps_steps
        eps_range = np.arange(eps_limit[0], eps_limit[1]+eps_step, eps_step)
        data_in_eps = np.ndarray(len(eps_range))
        if self.verbose:
            print ' Calculating number of points in eps bin:'
        for i_e in range(len(eps_range)):
            if self.verbose:
                print str(i_e+1)+' out of '+str(len(eps_range))
            # get number of distances that meet epsilon criteria
            data_in_eps[i_e] = np.sum(self.data_distances <= eps_range[i_e])
        # log and return results
        return np.log(eps_range), np.log(2*data_in_eps/(self.data_size[0]**2))


