extern crate nalgebra;
extern crate num;

use nalgebra as na;
use nalgebra::Iterable;
use std::ops::Mul;

#[derive(Debug, Clone, PartialEq)]
struct HilbertSpace {
    basis: Vec<PureState>,
    dim: usize,
}

#[derive(Debug, Clone, PartialEq)]
struct PureState {
    at: Vec<f64>,
    space: HilbertSpace,
    dim: usize,
}

#[derive(Debug, Clone, PartialEq)]
struct MixedState {
    at: na::DMatrix<f64>,
    space: Vec<HilbertSpace>,
    dim: (usize, usize),
}

impl HilbertSpace {
    fn new(dim: usize) -> HilbertSpace {
        HilbertSpace {
            basis: HilbertSpace::canonical_basis(dim)
                       .iter()
                       .map(|x| PureState::new(x))
                       .collect::<Vec<PureState>>(),
            dim: dim,
        }
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
}

impl PureState {
    fn new(at: &Vec<f64>) -> PureState {
        PureState {
            at: at.clone(),
            space: HilbertSpace::new(at.len()),
            dim: at.len(),
        }
    }
    fn to_mixed(&self) -> MixedState {
        na::outer(self, self)
    }
    // fn partial_trace(&self, space: usize) -> MixedState {
    //     self.to_mixed.partial_trace(space); 
    // }
}

impl MixedState {
    fn new(at: &na::DMatrix<f64>) -> MixedState {
        MixedState {
            at: at.clone(),
            space: vec![HilbertSpace::new(at.nrows())],
            dim: (at.nrows(), at.ncols()),
        }
    }
    fn partition(&mut self, partition: Vec<usize>) -> () {
        if self.space.len() > 1 {
            panic!("Space already partitioned");
        }
        self.space = self.space[0].partition(partition);
    }
    fn unpartition(&mut self) -> () {
        if self.space.len() < 2 {
            panic!("Space not partitioned");
        }
        let dim = self.space.iter().fold(0, |acc, x| acc + x.dim);
        self.space = vec![HilbertSpace::new(dim)];
    }
    fn partial_trace(self, basis: usize) -> MixedState {
        if self.space.len() < 2 {
            panic!("Space not partitioned");
        }
        if basis > self.space.len() {
            panic!("No such space");
        }
        self.space[basis].basis.iter().map(|vec| *vec*self* *vec).fold(0, |acc, x| acc+x)
    }
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
       pairs.map(|(x, y)| x * y).fold(0.0, |acc, x| acc+x)
    }
}

impl Mul<PureState> for MixedState {
    type Output = PureState;

    fn mul(self, rhs: PureState) -> PureState {
        let sli = rhs.at.clone();
        let dvec = self.at*na::DVector::from_slice(rhs.dim, sli.as_slice());
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
        let dvec = rhs.at*na::DVector::from_slice(self.dim, sli.as_slice());
        let mut vec = vec![];
        for elem in dvec.iter() {
          vec.push(*elem);
        }
        PureState::new(&vec)
    }
}

struct Channel {
    state: PureState,
}

fn main() {
    let x = HilbertSpace::new(5).partition(vec![2, 3]);
    println!("{:?}", x);
}

#[cfg(test)]
mod tests {
    use nalgebra as na;

    use super::{MixedState, PureState, HilbertSpace};

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
            assert_eq!(vec.at.len(), 2)
        }
    }

    #[test]
    fn partition() {
        let space = HilbertSpace::new(5);
        let partition = space.partition(vec![2, 3]);
        assert_eq!(partition[0].basis.len(), 1);
        assert_eq!(partition[1].basis.len(), 2);
    }


    #[test]
    fn mixed_state_new() {
        let mat = na::DMatrix::from_row_iter(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let state = MixedState::new(&mat);
        assert_eq!(state.at, mat);
        assert_eq!(state.dim, (state.at.nrows(), state.at.ncols()));
    }

    #[test]
    fn mixed_state_partition() {
        let mat = na::DMatrix::from_row_iter(3,
                                             3,
                                             vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let mut state = MixedState::new(&mat);
        state.partition(vec![1, 2]);
        assert!(state.space.len() > 1);
    }

    #[test]
    #[should_panic]
    fn mixed_state_double_partition_panic() {
        let mat = na::DMatrix::from_row_iter(3,
                                             3,
                                             vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let mut state = MixedState::new(&mat);
        state.partition(vec![1, 2]);
        state.partition(vec![2, 1]);
    }

    #[test]
    fn mixed_state_partition_unpartition() {
        let mat = na::DMatrix::from_row_iter(3,
                                             3,
                                             vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let mut state = MixedState::new(&mat);
        let old_state = state.clone();
        state.partition(vec![1, 2]);
        state.unpartition();
        assert_eq!(old_state, state);
    }

    #[test]
    fn pure_to_mixed() {
      let pure_state = PureState::new(&vec![1.0, 2.0, 3.0, 4.0]);
      let mixed_state = pure_state.to_mixed();
      let (a, b) = mixed_state.dim;
      assert_eq!(a, b);
      assert_eq!(a, pure_state.dim);
      assert_eq!(b, pure_state.dim);
    }

    #[test]
    fn pure_state_mul() {
      let pure_state_1 = PureState::new(&vec![1.0, 2.0, 3.0, 4.0]);
      let pure_state_2 = PureState::new(&vec![1.0, 2.0, 3.0, 4.0]);
      let out = pure_state_1*pure_state_2;
      assert_eq!(out, 1.0+4.0+9.0+16.0);
    }

    #[test]
    fn mixed_state_pure_state_mul() {
        let mat = na::DMatrix::from_row_iter(3,
                                             3,
                                             vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        let mixed_state = MixedState::new(&mat);
        let pure_state = PureState::new(&vec![1.0, 2.0, 3.0]);
        assert_eq!(mixed_state.clone()*pure_state.clone(), pure_state.clone());
        assert_eq!(mixed_state.clone()*pure_state.clone(), pure_state*mixed_state)
    }
}