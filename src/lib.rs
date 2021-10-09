/// This code is a Rust implementation of the
/// [Geometry Processing with Intrinsic Triangulations course](https://github.com/nmwsharp/intrinsic-triangulations-tutorial)
/// by [Nicholas Sharp](https://nmwsharp.com/), [Mark Gillespie](https://markjgillespie.com/), [Keenan Crane](http://keenan.is/here).
/// For a better understanding of the content, please watch the course and view the course repo.
use nalgebra::{DMatrix, DVector, Vector2};
use nalgebra_sparse::{convert::serial::convert_csr_dense, csr::CsrMatrix, CooMatrix};
use std::collections::{HashMap, VecDeque};

/// A useful way to describe the face tuple
/// Represents (face_index, face_side)
pub type FaceSide = (usize, usize);

pub trait NextSide {
    fn next(self) -> Self;
}

/// Returns the next face in a triangle
impl NextSide for FaceSide {
    fn next(self) -> Self {
        (self.0, (self.1 + 1) % 3)
    }
}

/// Returns the face glued to the supplied face
///
/// # Arguments
/// * `g` - A glue map stored as an |Fx3| DMatrix of FaceSides
/// * `fs` - A face side formatted as (face_index, face_side)
pub fn other(g: &DMatrix<FaceSide>, fs: FaceSide) -> FaceSide {
    g[fs]
}

/// Returns the number of faces in the face matrix
///
/// # Arguments
/// * `f` - An |Fx3| matrix of face indices
pub fn n_faces(f: &DMatrix<i64>) -> usize {
    f.shape().0
}

/// Returns the number of vertices in the face matrix
///
/// # Arguments
/// * `f` - An |Fx3| matrix of face indices
pub fn n_verts(f: &DMatrix<i64>) -> i64 {
    f.amax() + 1
}

/// Returns the area of a given face
///
/// # Arguments
/// * `l` - An |Fx3| matrix of face side lengths
/// * `f` - An |Fx3| matrix of face indices
pub fn face_area(l: &DMatrix<f64>, f: usize) -> f64 {
    let a = l[(f, 0)];
    let b = l[(f, 1)];
    let c = l[(f, 2)];
    let s = (a + b + c) / 2.0;
    let d = s * (s - a) * (s - b) * (s - c);
    d.sqrt()
}

/// Returns the surface area of a given mesh
///
/// # Arguments
/// * `f` - An |Fx3| matrix of face indices
/// * `l` - An |Fx3| matrix of face side lengths
pub fn surface_area(f: &DMatrix<i64>, l: &DMatrix<f64>) -> f64 {
    (0..n_faces(f)).map(|i| face_area(l, i)).sum()
}

/// Returns the opposite corner angle of a given face side
///
/// # Arguments
/// * `l` - An |Fx3| matrix of face side lengths
/// * `fs` - A face side formatted as (face_index, face_side)
pub fn opposite_corner_angle(l: &DMatrix<f64>, fs: FaceSide) -> f64 {
    let a = l[fs];
    let b = l[fs.next()];
    let c = l[fs.next().next()];
    let d = (b.powi(2) + c.powi(2) - a.powi(2)) / (2.0 * b * c);
    d.acos()
}
/// Returns the length of the diagonal of a given face side
///
/// # Arguments
/// * `g` - A glue map stored as an |Fx3| DMatrix of FaceSides
/// * `l` - An |Fx3| matrix of face side lengths
/// * `fs` - A face side formatted as (face_index, face_side)
pub fn diagonal_length(g: &DMatrix<FaceSide>, l: &DMatrix<f64>, fs: FaceSide) -> f64 {
    let fs_other = other(g, fs);
    let u = l[fs.next().next()];
    let v = l[fs_other.next()];
    let theta_a = opposite_corner_angle(l, fs.next());
    let theta_b = opposite_corner_angle(l, fs_other.next().next());
    let t = theta_a + theta_b;
    let d = u.powi(2) + v.powi(2) - 2.0 * u * v * t.cos();
    d.sqrt()
}

/// Returns a boolean of whether a given face is a delaunay triangulation.
///
/// # Arguments
/// * `g` - A glue map stored as an |Fx3| DMatrix of FaceSides
/// * `l` - An |Fx3| matrix of face side lengths
/// * `fs` - A face side formatted as (face_index, face_side)
pub fn is_delaunay(g: &DMatrix<FaceSide>, l: &DMatrix<f64>, fs: FaceSide) -> bool {
    let fs_other = other(g, fs);
    let theta_a = opposite_corner_angle(l, fs);
    let theta_b = opposite_corner_angle(l, fs_other);
    theta_a + theta_b <= std::f64::consts::PI + std::f64::EPSILON
}

/// Returns an |Fx3| matrix of f64 values representing the lengths of face edges.
///
/// # Arguments
/// * `v` - An |Fx3| matrix of vertex positions
/// * `f` - An |Fx3| matrix of face indices
pub fn build_edge_lengths(v: &DMatrix<f64>, f: &DMatrix<i64>) -> DMatrix<f64> {
    let mut l = DMatrix::from_element(n_faces(f), 3, 0.);
    for fi in 0..n_faces(f) {
        for s in 0..3 {
            let i = f[(fi, s)] as usize;
            let j = f[(fi, s).next()] as usize;
            let length = v.row(j) - v.row(i);
            let length = length.norm();
            l[(fi, s)] = length;
        }
    }
    l
}

/// Updates the gluing map so that fs1 and fs2 are glued together.
///
/// # Arguments
/// * `g` - A glue map stored as an |Fx3| DMatrix of FaceSides
/// * `fs1` - A face side formatted as (face_index, face_side)
/// * `fs2` - A face side formatted as (face_index, face_side)
pub fn glue_together(g: &mut DMatrix<FaceSide>, fs1: FaceSide, fs2: FaceSide) {
    g[fs1] = fs2;
    g[fs2] = fs1;
}

/// Tests whether the gluing map was successfully constructed.
///
/// # Arguments
/// * `f` - An |Fx3| matrix of face indices
/// * `g` - A glue map stored as an |Fx3| DMatrix of FaceSides
pub fn validate_gluing_map(f: &DMatrix<i64>, g: &DMatrix<FaceSide>) {
    for fi in 0..n_faces(f) {
        for s in 0..3 {
            let fs = (fi, s);
            let fs_other = other(g, fs);
            if fs == fs_other {
                panic!("gluing map points face-side to itself");
            }

            if fs != other(g, fs_other) {
                panic!("gluing map is not involution (applying it twice does not return the original face-side)")
            }
        }
    }
}

/// Returns an |Fx3| matrix of FaceSides representing the glue map of a mesh.
///
/// # Arguments
/// * `f` - An |Fx3| matrix of face indices
pub fn build_gluing_map(f: &DMatrix<i64>) -> DMatrix<FaceSide> {
    let mut s = (0..n_faces(f))
        .flat_map(|fi| {
            (0..3)
                .map(|si| {
                    let i = f[(fi, si)];
                    let j = f[(fi, si).next()];
                    (i.min(j), i.max(j), fi, si)
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    s.sort();

    let n_sides = 3 * n_faces(f);
    let mut g = DMatrix::from_element(n_faces(f), 3, (0, 0));
    for p in (0..n_sides).step_by(2) {
        if s[p].0 != s[p + 1].0 || s[p].1 != s[p + 1].1 {
            panic!("Problem building glue map. Is input closed & manifold?");
        }
        let sp0 = s[p];
        let fs0 = (sp0.2, sp0.3);
        let sp1 = s[p + 1];
        let fs1 = (sp1.2, sp1.3);
        glue_together(&mut g, fs0, fs1);
    }
    validate_gluing_map(&f, &g);
    g
}

/// Flip the edge of of a triangle. Returns a FaceSide that looks like (s0_face_index, 0).
///
/// # Arguments
/// * `f` - An |Fx3| matrix of face indices
/// * `g` - A glue map stored as an |Fx3| DMatrix of FaceSides
/// * `l` - An |Fx3| matrix of face side lengths
/// * `s0` - A face side formatted as (face_index, face_side)
pub fn flip_edge(
    f: &mut DMatrix<i64>,
    g: &mut DMatrix<FaceSide>,
    l: &mut DMatrix<f64>,
    s0: FaceSide,
) -> FaceSide {
    let s1 = other(g, s0);
    let s2 = s0.next();
    let s3 = s0.next().next();
    let s4 = s1.next();
    let s5 = s1.next().next();

    let s6 = other(g, s2);
    let s7 = other(g, s3);
    let s8 = other(g, s4);
    let s9 = other(g, s5);

    let v0 = f[s0];
    let v1 = f[s2];
    let v2 = f[s3];
    let v3 = f[s5];

    let f0 = s0.0;
    let f1 = s1.0;

    let l2 = l[s2];
    let l3 = l[s3];
    let l4 = l[s4];
    let l5 = l[s5];

    let new_length = diagonal_length(g, l, s0);
    f[(f0, 0)] = v3;
    f[(f0, 1)] = v2;
    f[(f0, 2)] = v0;
    f[(f1, 0)] = v2;
    f[(f1, 1)] = v3;
    f[(f1, 2)] = v1;

    let relabel = |s: FaceSide| match s {
        s if s == s2 => (f1, 2),
        s if s == s3 => (f0, 1),
        s if s == s4 => (f0, 2),
        s if s == s5 => (f1, 1),
        _ => s,
    };
    let s6 = relabel(s6);
    let s7 = relabel(s7);
    let s8 = relabel(s8);
    let s9 = relabel(s9);

    glue_together(g, (f0, 0), (f1, 0));
    glue_together(g, (f0, 1), s7);
    glue_together(g, (f0, 2), s8);
    glue_together(g, (f1, 1), s9);
    glue_together(g, (f1, 2), s6);

    l[(f0, 0)] = new_length;
    l[(f0, 1)] = l3;
    l[(f0, 2)] = l4;
    l[(f1, 0)] = new_length;
    l[(f1, 1)] = l5;
    l[(f1, 2)] = l2;

    (f0, 0)
}

/// Flip all the triangles in a matrix to delaunay.
///
/// # Arguments
/// * `f` - An |Fx3| matrix of face indices
/// * `g` - A glue map stored as an |Fx3| DMatrix of FaceSides
/// * `l` - An |Fx3| matrix of face side lengths
pub fn flip_to_delaunay(f: &mut DMatrix<i64>, g: &mut DMatrix<FaceSide>, l: &mut DMatrix<f64>) {
    let mut to_process: VecDeque<FaceSide> = VecDeque::new();
    for fi in 0..n_faces(f) {
        for s in 0..3 {
            to_process.push_back((fi, s))
        }
    }

    while let Some(fs) = to_process.pop_front() {
        if !is_delaunay(g, l, fs) {
            let fs = flip_edge(f, g, l, fs);
            to_process.push_back(fs.next());
            to_process.push_back(fs.next().next());
            to_process.push_back(other(g, fs).next());
            to_process.push_back(other(g, fs).next().next());
        }
    }
}

/// Returns a boolean that asserts a matrix has been converted to a delaunay representation.
///
/// # Arguments
/// * `f` - An |Fx3| matrix of face indices
/// * `g` - A glue map stored as an |Fx3| DMatrix of FaceSides
/// * `l` - An |Fx3| matrix of face side lengths
pub fn check_delaunay(f: &DMatrix<i64>, g: &DMatrix<FaceSide>, l: &DMatrix<f64>) -> bool {
    for i in 0..n_faces(f) {
        for s in 0..3 {
            if !is_delaunay(g, l, (i, s)) {
                return false;
            }
        }
    }
    true
}

/// Returns a cotan-Laplace matrix stored as a |V|x|V| CsrMatrix<f64>.
///
/// # Arguments
/// * `f` - An |Fx3| matrix of face indices
/// * `l` - An |Fx3| matrix of face side lengths
pub fn build_cotan_laplacian(f: &DMatrix<i64>, l: &DMatrix<f64>) -> CsrMatrix<f64> {
    let n = n_verts(f) as usize;
    let mut map: HashMap<FaceSide, f64> = HashMap::new();
    for fi in 0..n_faces(f) {
        for s in 0..3 {
            let fs = (fi, s);
            let i = f[fs] as usize;
            let j = f[fs.next()] as usize;
            let opp_theta = opposite_corner_angle(l, fs);
            let opp_cotan = 1.0 / opp_theta.tan();
            let cotan_weight = 0.5 * opp_cotan;
            let ij = map.entry((i, j)).or_insert(0.0);
            *ij -= cotan_weight;
            let ji = map.entry((j, i)).or_insert(0.0);
            *ji -= cotan_weight;
            let ii = map.entry((i, i)).or_insert(0.0);
            *ii += cotan_weight;
            let jj = map.entry((j, j)).or_insert(0.0);
            *jj += cotan_weight;
        }
    }
    let mut coo = CooMatrix::new(n, n);
    for (fs, value) in map {
        let (f, s) = fs;
        coo.push(f, s, value);
    }
    CsrMatrix::from(&coo)
}

/// Returns a lumped mass matrix stored as a |V|x|V| CsrMatrix<f64>.
///
/// # Arguments
/// * `f` - An |Fx3| matrix of face indices
/// * `l` - An |Fx3| matrix of face side lengths
pub fn build_lumped_mass(f: &DMatrix<i64>, l: &DMatrix<f64>) -> CsrMatrix<f64> {
    let n = n_verts(f) as usize;
    let mut map: HashMap<usize, f64> = HashMap::new();
    for fi in 0..n_faces(f) {
        let area = face_area(l, fi);
        for s in 0..3 {
            let i = f[(fi, s)] as usize;
            let mi = map.entry(i).or_insert(0.0);
            *mi += area / 3.0;
        }
    }
    let mut coo = CooMatrix::new(n, n);
    for (i, value) in map {
        coo.push(i, i, value);
    }
    CsrMatrix::from(&coo)
}

/// Returns a Vector2<f64> representing a corresponding edge in the supplied face.
///
/// # Arguments
/// * `f` - An |Fx3| matrix of face indices
/// * `fs` - A face side formatted as (face_index, face_side)
pub fn edge_in_face_basis(l: &DMatrix<f64>, fs: FaceSide) -> Vector2<f64> {
    let (f, s) = fs;
    let theta = opposite_corner_angle(l, (f, 1));

    let local_vert_positions = DMatrix::from_row_slice(
        3,
        2,
        &[
            0.0,
            0.0,
            l[(f, 0)],
            0.0,
            theta.cos() * l[(f, 2)],
            theta.sin() * l[(f, 2)],
        ],
    );
    let edge_vec = local_vert_positions.row((s + 1) % 3) - local_vert_positions.row(s);
    let v = edge_vec.as_slice();
    Vector2::new(v[0], v[1])
}

/// Returns a matrix representing a the gradient for each face.
///
/// # Arguments
/// * `f` - An |F|x3 matrix of face indices
/// * `l` - An |F|x3 matrix of face side lengths
/// * `x` - An |F|x2 matrix of gradient vectors per-face
pub fn evaluate_gradient_at_faces(
    f: &DMatrix<i64>,
    l: &DMatrix<f64>,
    x: &DVector<f64>,
) -> DMatrix<f64> {
    let mut grads = DMatrix::from_element(n_faces(f), 2, 0.0);
    for fi in 0..n_faces(f) {
        let mut face_grad = Vector2::new(0.0, 0.0);
        for s in 0..3 {
            let i = f[(fi, s)] as usize;
            let edge_vec = edge_in_face_basis(l, (fi, s).next());
            let edge_vec_rot = Vector2::new(-edge_vec.y, edge_vec.x);
            face_grad += x[i] * edge_vec_rot;
        }
        let area = face_area(l, fi);
        face_grad /= 2.0 * area;
        grads[(fi, 0)] = face_grad.x;
        grads[(fi, 1)] = face_grad.y;
    }
    grads
}

/// Returns a matrix representing a the divergence of the vector field.
///
/// # Arguments
/// * `f` - An |F|x3 matrix of face indices
/// * `l` - An |F|x3 matrix of face side lengths
/// * `v` - An |F|x2 matrix of vectors per-face
pub fn evaluate_divergence_at_vertices(
    f: &DMatrix<i64>,
    l: &DMatrix<f64>,
    v: &DMatrix<f64>,
) -> DVector<f64> {
    let mut divs = DVector::from_element(n_verts(f) as usize, 0.0);
    for fi in 0..n_faces(f) {
        let grad_vec = Vector2::new(v[(fi, 0)], v[(fi, 1)]);
        for s in 0..3 {
            let fs = (fi, s);
            let i = f[fs] as usize;
            let j = f[fs.next()] as usize;
            let edge_vec = edge_in_face_basis(l, fs);
            let opp_theta = opposite_corner_angle(l, fs);
            let opp_cotan = 1.0 / opp_theta.tan();
            let cotan_weight = 0.5 * opp_cotan;
            let div_contrib = cotan_weight * edge_vec.dot(&grad_vec);
            divs[i] += div_contrib;
            divs[j] -= div_contrib;
        }
    }
    divs
}

/// Returns a matrix representing the geodesic distance along the surface using the Heat Method.
///
/// # Arguments
/// * `f` - An |F|x3 matrix of face indices
/// * `l` - An |F|x3 matrix of face side lengths
/// * `source_vert` - The index of a vertex to use as a source
pub fn heat_method_distance_from_vertex(
    f: &DMatrix<i64>,
    l: &DMatrix<f64>,
    source_vert: i64,
) -> DVector<f64> {
    let ll = build_cotan_laplacian(f, l);
    let m = build_lumped_mass(f, l);

    let mean_edge_length = l.mean();
    let short_time = mean_edge_length.powi(2);

    let h = m + short_time * ll.clone();
    let mut init_rhs = DVector::from_element(n_verts(f) as usize, 0.0);
    init_rhs[source_vert as usize] = 1.0;
    let heat = convert_csr_dense(&h).lu().solve(&init_rhs).unwrap();
    let mut grads = evaluate_gradient_at_faces(f, l, &heat);
    let norms = (0..grads.shape().0)
        .map(|i| grads.row(i).norm())
        .collect::<Vec<_>>();

    for i in 0..grads.shape().0 {
        let g = grads.row(i) / norms[i];
        grads.set_row(i, &g);
    }

    let div = evaluate_divergence_at_vertices(f, l, &grads);
    let dist = ll.clone() + CsrMatrix::identity(ll.ncols()) * 1e-6;
    let mut dist = convert_csr_dense(&dist).lu().solve(&div).unwrap();
    let k = dist[source_vert as usize];
    for i in 0..dist.shape().0 {
        dist[i] -= k;
    }
    dist
}

/// Print some information about the given data.
///
/// # Arguments
/// * `f` - An |Fx3| matrix of face indices
/// * `g` - A glue map stored as an |Fx3| DMatrix of FaceSides
/// * `l` - An |Fx3| matrix of face side lengths
pub fn print_info(f: &DMatrix<i64>, g: &DMatrix<FaceSide>, l: &DMatrix<f64>) {
    println!("n_verts = {}", n_verts(f));
    println!("n_faces = {}", n_faces(f));
    println!("surface_area = {}", surface_area(f, l));
    println!("is Delaunay = {}", check_delaunay(f, g, l));
}