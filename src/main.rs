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
    println!("-- PEGASUS MODEL");
    let source_vert = 1669;
    let (models, _) = tobj::load_obj("pegasus.obj", &tobj::LoadOptions::default())
        .expect("Failed to load OBJ file");
    if let Some(model) = models.first() {
        let pos = Array::from_iter(model.mesh.positions.iter().map(|p| *p as f64));
        let vlen = pos.shape()[0];
        let v = pos.into_shape((vlen / 3, 3)).unwrap();
        let indices = Array::from_iter(model.mesh.indices.iter().map(|u| *u as Index));
        let flen = indices.shape()[0];
        let f = indices.into_shape((flen / 3, 3)).unwrap();
        let g = build_gluing_map(&f);
        let l = build_edge_lengths(&v, &f);

        println!("Initial mesh:");
        print_info(&f, &g, &l);
        let dist_before = heat_method_distance_from_vertex(&f, &l, source_vert);
        println!("{:?}", dist_before);

        println!("");

        let mut f_delaunay = f.clone();
        let mut g_delaunay = g.clone();
        let mut l_delaunay = l.clone();
        flip_to_delaunay(&mut f_delaunay, &mut g_delaunay, &mut l_delaunay);
        println!("After Delaunay flips:");
        print_info(&f_delaunay, &g_delaunay, &l_delaunay);
        let dist_after = heat_method_distance_from_vertex(&f_delaunay, &l_delaunay, source_vert);
        let mut heatcsv = String::from("dist_before,dist_after,v0,v1,v2\n");
        for i in 0..v.shape()[0] {
            let data = format!(
                "{},{},{},{},{}\n",
                dist_before[i],
                dist_after[i],
                v[[i, 0]],
                v[[i, 1]],
                v[[i, 2]]
            );
            heatcsv.push_str(&data);
        }
        let mut facecsv = String::from("f0,f1,f2\n");
        for i in 0..f.shape()[0] {
            let data = format!("{},{},{}\n", f[[i, 0]], f[[i, 1]], f[[i, 2]]);
            facecsv.push_str(&data);
        }
        std::fs::write("distance.csv", &heatcsv).unwrap();
        std::fs::write("faces.csv", &facecsv).unwrap();
    }
}
