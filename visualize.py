import polyscope as ps
import pandas as pd

dists = pd.read_csv("distance.csv")
dists_before = dists.iloc[:, 0].to_numpy()
dists_after = dists.iloc[:, 1].to_numpy()

V = dists[["v0", "v1", "v2"]].to_numpy()
F = pd.read_csv("faces.csv").to_numpy()

ps.init()
ps_mesh = ps.register_surface_mesh("Pegasus", V, F)
ps_mesh.add_distance_quantity(
    "distance on initial triangulation", dists_before, enabled=True
)
ps_mesh.add_distance_quantity("distance after Delaunay flips", dists_after)
ps.show()
