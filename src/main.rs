extern crate nalgebra;
extern crate num;

use nalgebra as na;
use nalgebra::Iterable;
use std::ops::{Add, Mul};

#[derive(Debug, Clone, PartialEq)]
struct VectorPartition(Vec<Vec<f64>>, Vec<HilbertSpace>, Vec<usize>);
#[derive(Debug, Clone, PartialEq)]
struct MatrixPartition(Vec<na::DMatrix<f64>>, Vec<HilbertSpace>, Vec<usize>);

impl MatrixPartition {
    fn new(matrices: Vec<na::DMatrix<f64>>) -> Self {
        let dimensions = matrices.clone().iter().map(|mat| mat.nrows()).collect::<Vec<usize>>();
        let spaces = dimensions.clone().iter().map(|dim| HilbertSpace::new(*dim)).collect::<Vec<HilbertSpace>>(); 
        MatrixPartition(matrices, spaces, dimensions)
    }
    fn nil() -> Self {
        MatrixPartition(vec![na::DMatrix::new_zeros(1, 1)], vec![HilbertSpace::nil()], vec![])
    }
}

impl VectorPartition {
    fn new(vectors: Vec<Vec<f64>>) -> Self {
        let dimensions = vectors.clone().iter().map(|vec| vec.len()).collect::<Vec<usize>>();
        let spaces = dimensions.clone().iter().map(|dim| HilbertSpace::new(*dim)).collect::<Vec<HilbertSpace>>(); 
        VectorPartition(vectors, spaces, dimensions)
    }
    fn nil() -> Self {
        VectorPartition(vec![vec![]], vec![HilbertSpace::nil()], vec![])
    }
}

#[derive(Debug, Clone, PartialEq)]
struct HilbertSpace {
    basis: Vec<Vec<f64>>,
    dim: usize,
}

impl HilbertSpace {
    fn new(dim: usize) -> HilbertSpace {
        HilbertSpace {
            basis: HilbertSpace::canonical_basis(dim),
            dim: dim,
        }
    }
    fn nil() -> HilbertSpace {
        HilbertSpace::new(0)
    }

    fn partition(&self, partition: Vec<usize>) -> Vec<HilbertSpace> {
        if partition.iter().fold(0, |ac, x| ac + x) != self.dim {
            panic!("Not a valid partition");
        } else {
            let mut ret = vec![];
            for dim in partition {
                ret.push(HilbertSpace::new(dim));
            }
            ret
        }
    }

    fn canonical_basis(dim: usize) -> Vec<Vec<f64>> {
        let mut ret_1: Vec<Vec<f64>> = vec![];
        let mut ret_2: Vec<f64> = vec![];
        for i in 1..dim {
            for j in 1..dim {
                if i == j {
                    ret_2.push(1.0);
                } else {
                    ret_2.push(0.0);
                }
            }
            ret_1.push(ret_2.clone());
            ret_2 = vec![];
        }
        ret_1
    }
    
    fn zero(dim: usize) -> Vec<f64> {
        let mut ret = vec![];
        for _ in 1..dim {
            ret.push(0.0);
        }
        ret
    }

    fn ones(dim: usize) -> Vec<f64> {
        let mut ret = vec![];
        for _ in 1..dim {
            ret.push(1.0);
        }
        ret
    }
}
#[derive(Debug, Clone, PartialEq)]
struct PureState {
    at: Vec<f64>,
    space: HilbertSpace,
    dim: usize,
    partition: MatrixPartition,
}

#[derive(Debug, Clone, PartialEq)]
struct MixedState {
    at: na::DMatrix<f64>,
    space: HilbertSpace,
    dim: usize,
    partition: MatrixPartition,
}


impl PureState {
    fn new(at: &Vec<f64>) -> PureState {
        PureState {
            at: at.clone(),
            space: HilbertSpace::new(at.len()),
            dim: at.len(),
            partition: MatrixPartition::nil(),
        }
    }
    fn nil() -> PureState {
        PureState::new(&vec![]) 
    }
    fn to_mixed(&self) -> MixedState {
        na::outer(self, self)
    }
    // fn partition(&self, partition: Vec<usize>) -> PureState {
    //     let partition = 
    //     PureState {
    //         at: self.at.clone(),
    //         space: self.space.clone(), 
    //         dim: self.dim,
    //         partition:
    //     }
    // }
}

impl MixedState {
    fn new(at: &na::DMatrix<f64>) -> MixedState {
        if at.nrows() != at.ncols() {
            panic!("Require a square matrix");
        }
        MixedState {
            at: at.clone(),
            space: HilbertSpace::new(at.nrows()),
            dim: at.nrows(),
            partition: MatrixPartition::nil(),
        }
    }
    fn nil() -> MixedState {
        MixedState::new(&na::DMatrix::new_zeros(0, 0))
    }
    // fn partition(&mut self, partition: Vec<usize>) -> () {
    //     if self.space.len() > 1 {
    //         panic!("Space already partitioned");
    //     }
    //     self.space = self.space[0].partition(partition);
    // }
    // fn unpartition(&mut self) -> () {
    //     if self.space.len() < 2 {
    //         panic!("Space not partitioned");
    //     }
    //     let dim = self.space.iter().fold(0, |acc, x| acc + x.dim);
    //     self.space = vec![HilbertSpace::new(dim)];
    // }
    // fn partial_trace(self, basis: usize) -> MixedState {
    //     if self.space.len() < 2 {
    //         panic!("Space not partitioned");
    //     }
    //     if basis > self.space.len() {
    //         panic!("No such space");
    //     }
    // }
}

impl na::Outer for PureState {
    type OuterProductType = MixedState;
    fn outer(&self, other: &Self) -> Self::OuterProductType {
        MixedState::new(&na::DMatrix::from_fn(self.at.len(),
                                              other.at.len(),
                                              |i, j| self.at[i] * other.at[j]))
    }
}

impl Mul for PureState {
    type Output = f64;

    fn mul(self, rhs: PureState) -> f64 {
        let pairs = self.at.iter().zip(rhs.at.iter());
        pairs.map(|(x, y)| x * y).fold(0.0, |acc, x| acc + x)
    }
}

impl Mul<PureState> for MixedState {
    type Output = PureState;

    fn mul(self, rhs: PureState) -> PureState {
        let sli = rhs.at.clone();
        let dvec = self.at * na::DVector::from_slice(rhs.dim, sli.as_slice());
        let mut vec = vec![];
        for elem in dvec.iter() {
            vec.push(*elem);
        }
        PureState::new(&vec)
    }
}

impl Mul<MixedState> for PureState {
    type Output = PureState;

    fn mul(self, rhs: MixedState) -> PureState {
        let sli = self.at.clone();
        let dvec = rhs.at * na::DVector::from_slice(self.dim, sli.as_slice());
        let mut vec = vec![];
        for elem in dvec.iter() {
            vec.push(*elem);
        }
        PureState::new(&vec)
    }
}

impl Add<MixedState> for MixedState {
    type Output = MixedState;

    fn add(self, rhs: MixedState) -> MixedState {
        let ret = self.at.clone() + rhs.at.clone();
        MixedState::new(&ret)
    }
}

impl Add<PureState> for PureState {
    type Output = PureState;

    fn add(self, rhs: PureState) -> PureState {
        let sum = self.at.iter().zip(rhs.at.iter()).map(|(a, b)| a + b).collect::<Vec<_>>();
        PureState::new(&sum)
    }
}

fn main() {
        let x = PureState::new(&vec![1.0, 2.0]);
        // let y: MixedState = na::outer(&x, &x);
        println!("{:?}", x);
}

#[cfg(test)]
mod tests {
    use nalgebra as na;

    use super::{MixedState, PureState, HilbertSpace, MatrixPartition, VectorPartition};

    #[test]
    fn outer_test() {
        let x = PureState::new(&vec![1.0, 2.0]);
        let y: MixedState = na::outer(&x, &x);
        assert_eq!(y.at,
                   na::DMatrix::from_row_iter(2, 2, vec![1.0, 2.0, 2.0, 4.0]));
    }
    #[test]
    fn canonical() {
        for vec in HilbertSpace::new(3).basis.iter() {
            assert_eq!(vec.len(), 2)
        }
    }
    #[test]
    fn mixed_state_new() {
        let mat = na::DMatrix::from_row_iter(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let state = MixedState::new(&mat);
        assert_eq!(state.at, mat);
        assert_eq!(state.dim, state.at.nrows());
    }

    #[test]
    #[should_panic]
    fn non_square_matrix() {
        let mat = na::DMatrix::from_row_iter(4, 1, vec![1.0, 2.0, 3.0, 4.0]);
        let state = MixedState::new(&mat);
        assert_eq!(state.at, mat);
        assert_eq!(state.dim, state.at.nrows());
    }

    #[test]
    fn vector_partition_new() {
        let j = vec![vec![1.0, 2.0], vec![2.0, 1.0]];
        let x = VectorPartition::new(j.clone());
        let VectorPartition(i, _, _) = x;
        assert_eq!(j, i);
    }

    #[test]
    fn matrix_partition_new() {
        let mut mats = vec![];
        for _ in 1..10 {
            mats.push(na::DMatrix::new_random(3, 3));
        }
        let x = MatrixPartition::new(mats.clone());
        let MatrixPartition(x, y, z) = x;
        assert_eq!(x, mats);
    }

    // #[test]
    // #[should_panic]
    // fn pure_state_partition_wrong_dims() {

    // }

    // #[test]
    // fn pure_state_partition_new() {
    //     let pure_state = PureState::new(&vec![1.0, 2.0, 3.0, 4.0]);
    //     let new = pure_state.partition(vec![2, 2]);
    // }



    #[test]
    fn pure_to_mixed() {
        let pure_state = PureState::new(&vec![1.0, 2.0, 3.0, 4.0]);
        let mixed_state = pure_state.to_mixed();
        assert_eq!(mixed_state.dim, pure_state.dim);
    }

    #[test]
    fn pure_state_mul() {
        let pure_state_1 = PureState::new(&vec![1.0, 2.0, 3.0, 4.0]);
        let pure_state_2 = PureState::new(&vec![1.0, 2.0, 3.0, 4.0]);
        let out = pure_state_1 * pure_state_2;
        assert_eq!(out, 1.0 + 4.0 + 9.0 + 16.0);
    }

    #[test]
    fn mixed_state_pure_state_mul() {
        let mat = na::DMatrix::from_row_iter(3,
                                             3,
                                             vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        let mixed_state = MixedState::new(&mat);
        let pure_state = PureState::new(&vec![1.0, 2.0, 3.0]);
        assert_eq!(mixed_state.clone() * pure_state.clone(), pure_state.clone());
        assert_eq!(mixed_state.clone() * pure_state.clone(),
                   pure_state * mixed_state)
    }
}
