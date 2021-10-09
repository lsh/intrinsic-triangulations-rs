import potpourri3d as pp3d
import polyscope as ps
import pandas as pd

dists = pd.read_csv("distance.csv")
dists_before = dists.iloc[:, 0].to_numpy()
dists_after = dists.iloc[:, 1].to_numpy()
(V, F) = pp3d.read_mesh("pegasus.obj")

ps.init()
ps_mesh = ps.register_surface_mesh("Pegasus", V, F)
ps_mesh.add_distance_quantity(
    "distance on initial triangulation", dists_before, enabled=True
)
ps_mesh.add_distance_quantity("distance after Delaunay flips", dists_after)
ps.show()
