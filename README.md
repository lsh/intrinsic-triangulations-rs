# Intrinsic Triangulations in Rust

In this repo is code I wrote following along with the [Nicholas Sharp](https://nmwsharp.com/), [Mark Gillespie](https://markjgillespie.com/), [Keenan Crane](http://keenan.is/here)'s [course on geometry processing with intrinsic triangulations](https://www.youtube.com/watch?v=gcRDdYrgOhg). Also check out the [course repo](https://github.com/nmwsharp/intrinsic-triangulations-tutorial), which has more detailed comments.

The binary target of this project generates two files `distance.csv` and `faces.csv`, which I use to visualize the results of the pegasus heat distance using [Polyscope](https://polyscope.run/).
In order to generate the csvs, run:
```shell
cargo run --release
```
To visualize the results (requires potpourri3d and polyscope), run:
```shell
python visualize.py
```