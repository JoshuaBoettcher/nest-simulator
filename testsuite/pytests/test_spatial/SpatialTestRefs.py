import nest
import numpy as np


class SpatialTestRefs:
    """
    Provides common fixtures for source and target references used by the spatial tests.

         Sources                      Targets
      2  7 12 17 22    	          28 33 38 43 48
      3  8 13 18 23		          29 34	39 44 49
      4	 9 14 19 24		          30 35	40 45 50
      5	10 15 20 25		          31 36	41 46 51
      6	11 16 21 26		          32 37	42 47 52
    """

    src_layer_ref = np.array(
        [[1., -0.5, 0.5], [2., -0.5, 0.25], [3., -0.5, 0.], [4., -0.5, -0.25], [5., -0.5, -0.5], [6., -0.25, 0.5],
         [7., -0.25, 0.25], [8., -0.25, 0.], [9., -0.25, -0.25], [10., -0.25, -0.5], [11., 0., 0.5],
         [12., 0., 0.25], [13., 0., 0.], [14., 0., -0.25], [15., 0., -0.5], [16., 0.25, 0.5], [17., 0.25, 0.25],
         [18., 0.25, 0.], [19., 0.25, -0.25], [20., 0.25, -0.5], [21., 0.5, 0.5], [22., 0.5, 0.25], [23., 0.5, 0.],
         [24., 0.5, -0.25], [25., 0.5, -0.5]])

    target_layer_ref = np.array(
        [[26.0, -0.5, 0.5], [27, -0.5, 0.25], [28, -0.5, 0], [29, -0.5, -0.25], [30, -0.5, -0.5], [31, -0.25, 0.5],
         [32, -0.25, 0.25], [33, -0.25, 0], [34, -0.25, -0.25], [35, -0.25, -0.5], [36, 0, 0.5], [37, 0, 0.25],
         [38, 0, 0], [39, 0, -0.25], [40, 0, -0.5], [41, 0.25, 0.5], [42, 0.25, 0.25], [43, 0.25, 0],
         [44, 0.25, -0.25], [45, 0.25, -0.5], [46, 0.5, 0.5], [47, 0.5, 0.25], [48, 0.5, 0], [49, 0.5, -0.25],
         [50, 0.5, -0.5]])

    def map_connections(self, connections, src_layer, target_layer):
        mapping = []
        for node in connections:
            displacement = nest.Displacement(target_layer[node.target % (len(target_layer) + 1)],
                                             src_layer[node.source - 1])
            mapping += [[node.source, node.target, node.weight, node.delay, displacement[0][0], displacement[0][1]]]

        return np.sort(np.array(mapping), axis=0)
