use intrinsic_triangles::*;
use nalgebra::DMatrix;
fn main() {
    let source_vert = 0;
    let vdata = vec![0., 5., 0., 0., 1., -3., -4., 0., 0., 0., 1., 3., 4., 0., 0.];
    let v = DMatrix::from_row_slice(vdata.len() / 3, 3, vdata.as_slice());

    let fdata = vec![0, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 1, 1, 4, 2, 2, 4, 3];
    let f = DMatrix::from_row_slice(fdata.len() / 3, 3, fdata.as_slice());
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
    println!(
        "dist before: {}",
        heat_method_distance_from_vertex(&f, &l, source_vert)
    );
    println!(
        "dist after: {}",
        heat_method_distance_from_vertex(&f_delaunay, &l_delaunay, source_vert)
    );
    println!("");

    println!("-- PEGASUS MODEL");

    let source_vert = 1669;
    let (models, _) = tobj::load_obj("pegasus.obj", &tobj::LoadOptions::default())
        .expect("Failed to load OBJ file");
    if let Some(model) = models.first() {
        let pos = model
            .mesh
            .positions
            .iter()
            .map(|p| *p as f64)
            .collect::<Vec<_>>();
        let v = DMatrix::from_row_slice(pos.len() / 3, 3, pos.as_slice());
        let indices = model
            .mesh
            .indices
            .iter()
            .map(|i| *i as i64)
            .collect::<Vec<_>>();
        let f = DMatrix::from_row_slice(indices.len() / 3, 3, indices.as_slice());
        let g = build_gluing_map(&f);
        let l = build_edge_lengths(&v, &f);

        println!("Initial mesh:");
        print_info(&f, &g, &l);

        println!("");

        let mut f_delaunay = f.clone();
        let mut g_delaunay = g.clone();
        let mut l_delaunay = l.clone();
        flip_to_delaunay(&mut f_delaunay, &mut g_delaunay, &mut l_delaunay);
        println!("After Delaunay flips:");
        print_info(&f_delaunay, &g_delaunay, &l_delaunay);
        let dist_before = heat_method_distance_from_vertex(&f, &l, source_vert);
        let dist_after = heat_method_distance_from_vertex(&f_delaunay, &l_delaunay, source_vert);
        let dist_before_slice = dist_before.as_slice();
        let dist_after_slice = dist_after.as_slice();
        let mut heatcsv = String::from("dist_before,dist_after\n");
        for i in 0..dist_after_slice.len() {
            let data = format!("{},{}\n", dist_before_slice[i], dist_after_slice[i]);
            heatcsv.push_str(&data);
        }
        std::fs::write("distance.csv", &heatcsv).unwrap();
    }
}
