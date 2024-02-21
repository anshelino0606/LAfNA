use std::ops::{Add, Div, Mul, Sub};

trait Tensor<T> {
    fn shape(&self) -> &[usize];
    fn size(&self) -> usize;
}

struct Vector<T> {
    data: Vec<T>,
}

impl<T> Tensor<T> for Vector<T> {
    fn shape(&self) -> &[usize] {
        static SHAPE: [usize; 1] = [1];
        &SHAPE
    }

    fn size(&self) -> usize {
        self.data.len()
    }
}

struct Matrix<T> {
    data: Vec<Vec<T>>,
    rows: usize,
    cols: usize,
}

impl<T> Matrix<T>
where
    T: Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Clone + Default,
{
    fn add(&self, other: &Matrix<T>) -> Result<Matrix<T>, &'static str> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err("Matrices dimensions do not match");
        }

        let mut result = vec![vec![T::default(); self.cols]; self.rows];

        for i in 0..self.rows {
            for j in 0..self.cols {
                result[i][j] = self.data[i][j].clone() + other.data[i][j].clone();
            }
        }

        Ok(Matrix {
            data: result,
            rows: self.rows,
            cols: self.cols,
        })
    }

    fn subtract(&self, other: &Matrix<T>) -> Result<Matrix<T>, &'static str> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err("Matrices dimensions do not match");
        }

        let mut result = vec![vec![T::default(); self.cols]; self.rows];

        for i in 0..self.rows {
            for j in 0..self.cols {
                result[i][j] = self.data[i][j].clone() - other.data[i][j].clone();
            }
        }

        Ok(Matrix {
            data: result,
            rows: self.rows,
            cols: self.cols,
        })

    }

    fn multiply_scalar(&self, scalar: T) -> Matrix<T> {
        let mut result = vec![vec![T::default(); self.cols]; self.rows];

        for i in 0..self.rows {
            for j in 0..self.cols {
                result[i][j] = self.data[i][j].clone() * scalar.clone();
            }
        }

        Matrix {
            data: result,
            rows: self.rows,
            cols: self.cols,
        }
    }
}

impl<T> Tensor<T> for Matrix<T> {
    fn shape(&self) -> &[usize] {
        static SHAPE: [usize; 2] = [self.rows, self.cols];
        &SHAPE
    }

    fn size(&self) -> usize {
        self.rows * self.cols
    }
}

trait Arithmetic<T> {
    ///
    /// Adds 2 tensors T together (of equal shapes)
    fn add(&self, other: &T) -> T;
    ///
    /// Adds a scalar to every element of tensor
    fn addScalarTo(&self, scalar: f64) -> T;
    ///
    /// Directly adds a scalar to every element of tensor
    fn addScalar(&self, scalar: f64);
    ///
    /// Subtracts two tensors one from another
    fn subtract(&self, other: &T) -> T;
    ///
    /// Subtracts a scalar from every element of a tensor
    fn subtractScalarFrom(&self, scalar: f64) -> T;
    ///
    /// Directly subtracts a scalar from every element of a tensor
    fn subtractScalar(&self, scalar: f64);
    ///
    /// Multiplies a scalar by every element of tensor
    fn multiplyScalarBy(&self, scalar: f64) -> T;
    ///
    /// Directly multiplies a scalar by every element of tensor
    fn multiplyScalar(&self, scalar: f64);
    ///
    /// Divides a scalar by every element of tensor
    fn divideScalarBy(&self, scalar: f64) -> T;
    ///
    /// Directly divides a scalar by every element of tensor
    fn divideScalar(&self, scalar: f64);
    ///
    /// Raises every element to some power `pow`
    fn toPower(&self, pow: f64);
}

impl<T> Arithmetic<Matrix<T>> for Matrix<T> {
    fn add(&self, other: &Matrix<T>) -> Matrix<T> {
        self.add(other)
            .unwrap_or_else(|_| panic!("Matrix addition error: dimensions do not match"))
    }

    fn addScalarTo(&self, scalar: f64) -> Matrix<T> {
        todo!()
    }

    fn addScalar(&self, scalar: f64) {
        // Implement direct scalar addition to self...
    }

    fn subtract(&self, other: &Matrix<T>) -> Matrix<T> {
        self.subtract(other)
            .unwrap_or_else(|_| panic!("Matrix subtraction error: dimensions do not match"))
    }


    fn subtractScalarFrom(&self, scalar: f64) -> Matrix<T> {
        todo!()
    }

    fn subtractScalar(&self, scalar: f64) {
        todo!()
    }

    fn multiplyScalarBy(&self, scalar: f64) -> Matrix<T> {
        self.multiply_scalar(scalar)
    }

    fn multiplyScalar(&self, scalar: f64) {
        todo!()
    }

    fn divideScalarBy(&self, scalar: f64) -> Matrix<T> {
        todo!()
    }

    fn divideScalar(&self, scalar: f64) {
        todo!()
    }

    fn toPower(&self, pow: f64) {
        todo!()
    }
}

fn main() {}
