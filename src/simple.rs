use intrinsic_triangles::*;
use ndarray::prelude::*;

fn main() {
    let source_vert = 0;
    let v = array![
        [0., 5., 0.],
        [0., 1., -3.],
        [-4., 0., 0.],
        [0., 1., 3.],
        [4., 0., 0.]
    ];

    let f = array![
        [0, 1, 2],
        [0, 2, 3],
        [0, 3, 4],
        [0, 4, 1],
        [1, 4, 2],
        [2, 4, 3]
    ];
    let g = build_gluing_map(&f);
    let l = build_edge_lengths(&v, &f);

    println!("-- TEST DATA");
    println!("Initial mesh:");
    print_info(&f, &g, &l);

    println!("");

    let mut f_delaunay = f.clone();
    let mut g_delaunay = g.clone();
    let mut l_delaunay = l.clone();
    flip_to_delaunay(&mut f_delaunay, &mut g_delaunay, &mut l_delaunay);
    println!("After Delaunay flips:");
    print_info(&f_delaunay, &g_delaunay, &l_delaunay);
    println!("");
    let dist_before = heat_method_distance_from_vertex(&f, &l, source_vert);
    println!("{:?}", dist_before);
    let dist_after = heat_method_distance_from_vertex(&f_delaunay, &l_delaunay, source_vert);
    println!("{:?}", dist_after);
}
