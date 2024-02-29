extern crate core;

use core::f64;
use std::fs::File;
use std::{fmt, io};
use std::io::{BufRead, BufReader};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};
use std::path::Path;
use std::str::FromStr;

struct Vector {
    data: Vec<f64>,
    len: usize
}

#[derive(Debug, Clone)]
struct Matrix {
    data: Vec<Vec<f64>>,
    rows: usize,
    cols: usize,
}

impl Matrix
{
    // Constructor for a new Matrix with given rows and columns, initialized to default for type T
    fn new(rows: usize, cols: usize) -> Self {
        let default_value = f64::default();
        Matrix {
            data: vec![vec![default_value; cols]; rows],
            rows,
            cols,
        }
    }

    // Create an identity matrix of size dim
    // Note: This makes sense only for numeric types that can represent 1 and 0
    fn identity(dim: usize) -> Self
    {
        let mut matrix = Self::new(dim, dim);
        for i in 0..dim {
            matrix.data[i][i] = f64::from(1);
        }
        matrix
    }

    // Create a zero matrix with given dimensions
    // This uses T::default() and assumes that the default value for T represents "zero"
    fn zero(rows: usize, cols: usize) -> Self {
        Self::new(rows, cols)
    }

    // Prints the matrix
    fn print(&self)
    {
        for row in &self.data {
            println!("{:?}", row);
        }
    }

    // Method to add two matrices
    fn add(&self, other: &Matrix) -> Option<Matrix> {
        if self.rows != other.rows || self.cols != other.cols {
            return None;
        }

        let mut result = Matrix::new(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[i][j] = self.data[i][j].clone() + other.data[i][j].clone();
            }
        }
        Some(result)
    }

    // Method to subtract two matrices
    fn subtract(&self, other: &Matrix) -> Option<Matrix> {
        if self.rows != other.rows || self.cols != other.cols {
            return None;
        }

        let mut result = Matrix::new(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[i][j] = self.data[i][j].clone() - other.data[i][j].clone();
            }
        }
        Some(result)
    }

    // Method to multiply each element of a matrix by a scalar
    // Assumes that T can be multiplied by T and returns T
    fn scalar_multiply(&mut self, scalar: f64) {
        for i in 0..self.rows {
            for j in 0..self.cols {
                self.data[i][j] = self.data[i][j].clone() * scalar.clone();
            }
        }
    }

    // Method to multiply two matries
    fn multiply(&self, other: &Matrix) -> Option<Matrix> {
        if self.cols != other.rows {
            return None; // Incompatible dimensions
        }

        let mut result = Matrix::new(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                result.data[i][j] = (0..self.cols)
                    .map(|k| self.data[i][k].clone() * other.data[k][j].clone())
                    .sum();
            }
        }
        Some(result)
    }

    // Swap two rows by indices
    fn swap_rows(&mut self, row1: usize, row2: usize) {
        if row1 < self.rows && row2 < self.rows {
            self.data.swap(row1, row2);
        }
    }

    // Swap two columns by indices
    fn swap_cols(&mut self, col1: usize, col2: usize) {
        if col1 < self.cols && col2 < self.cols {
            for row in self.data.iter_mut() {
                row.swap(col1, col2);
            }
        }
    }

    // Multiply a row by a scalar
    fn multiply_row_by_scalar(&mut self, row_index: usize, scalar: f64) {
        if row_index < self.rows {
            for j in 0..self.cols {
                self.data[row_index][j] = self.data[row_index][j].clone() * scalar.clone();
            }
        }
    }

    // Multiply a column by a scalar
    fn multiply_col_by_scalar(&mut self, col_index: usize, scalar: f64) {
        if col_index < self.cols {
            for i in 0..self.rows {
                self.data[i][col_index] = self.data[i][col_index].clone() * scalar.clone();
            }
        }
    }

    fn gaussian_elimination(&mut self) -> i32 {
        let n = self.rows;
        let mut sign = 1; // Keep track of the sign of the determinant
        let epsilon = 1e-10; // Arbitrarily chosen small value

        for i in 0..n {
            // Find the pivot row based on maximum absolute value
            let mut max_row = i;
            for k in i + 1..n {
                if self.data[k][i].abs() > self.data[max_row][i].abs() {
                    max_row = k;
                }
            }

            // If the pivot is effectively zero, skip the current column
            if self.data[max_row][i].abs() < epsilon {
                continue;
            }

            // Swap if the pivot row is different from the current row
            if i != max_row {
                self.swap_rows(i, max_row);
                sign = -sign; // A row swap changes the sign of the determinant
            }

            // Perform elimination below the pivot
            for j in i + 1..n {
                let factor = self.data[j][i] / self.data[i][i];
                for k in i..self.cols {
                    self.data[j][k] -= factor * self.data[i][k];
                }
            }
        }

        sign
    }
}


impl Matrix
{
    fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, io::Error> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        let mut data: Vec<Vec<f64>> = Vec::new();

        for line in reader.lines() {
            let row_data = line?
                .split_whitespace()
                .map(|s| s.parse().expect("Could not parse number"))
                .collect::<Vec<f64>>();
            data.push(row_data);
        }

        let rows = data.len();
        // Assuming the matrix is not empty and is well-formed (all rows have the same length)
        let cols = if rows > 0 { data[0].len() } else { 0 };

        Ok(Matrix { data, rows, cols })
    }
}

impl Vector {
    fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, io::Error> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        let data: Vec<f64> = reader.lines()
            .map(|line| line.expect("Could not read line")
                .parse::<f64>().expect("Could not parse number"))
            .collect();

        let len = data.len();
        Ok(Vector { data, len })
    }
}

mod numerical_methods {
    use super::Matrix; // Assuming Matrix is defined in the parent module

    pub fn gaussian_elimination(matrix: &mut Matrix) -> Result<f64, &'static str> {
        let n = matrix.rows;
        let mut sign = 1;
        let epsilon = 1e-10;
        let mut det = 1.0;

        for i in 0..n {
            // Find the pivot row based on maximum absolute value
            let mut max_row = i;
            for k in i + 1..n {
                if matrix.data[k][i].abs() > matrix.data[max_row][i].abs() {
                    max_row = k;
                }
            }

            if matrix.data[max_row][i].abs() < epsilon {
                return Err("Matrix is singular or nearly singular");
            }

            // Swap if the pivot row is different from the current row
            if i != max_row {
                matrix.swap_rows(i, max_row);
                sign = -sign; // A row swap changes the sign of the determinant
            }

            // Perform elimination below the pivot
            for j in i + 1..n {
                let factor = matrix.data[j][i] / matrix.data[i][i];
                for k in i..matrix.cols {
                    matrix.data[j][k] -= factor * matrix.data[i][k];
                }
                matrix.data[j][i] = 0.0; // Explicitly setting to zero for clarity
            }

            det *= matrix.data[i][i];
        }

        Ok(det * sign as f64)
    }

    pub fn solve(matrix: &Matrix, b: Vec<f64>) -> Result<Vec<f64>, &'static str> {
        if matrix.rows != b.len() {
            return Err("Matrix and vector dimensions do not match");
        }

        // Clone matrix and augment it with b
        let mut aug_matrix = matrix.clone();
        for i in 0..aug_matrix.rows {
            aug_matrix.data[i].push(b[i]);
        }
        aug_matrix.cols += 1;

        // Apply Gaussian elimination
        if gaussian_elimination(&mut aug_matrix).is_err() {
            return Err("Matrix is singular or nearly singular");
        }

        // Back substitution
        let mut x = vec![0.0; matrix.rows];
        for i in (0..matrix.rows).rev() {
            x[i] = aug_matrix.data[i][matrix.cols] / aug_matrix.data[i][i];
            for k in (0..i).rev() {
                aug_matrix.data[k][matrix.cols] -= aug_matrix.data[k][i] * x[i];
            }
        }

        Ok(x)
    }

    pub fn invert(matrix: &Matrix) -> Result<Matrix, &'static str> {
        if matrix.rows != matrix.cols {
            return Err("Matrix must be square to compute its inverse");
        }

        // Clone matrix and augment it with the identity matrix
        let mut aug_matrix = matrix.clone();
        for i in 0..aug_matrix.rows {
            for j in 0..matrix.rows {
                aug_matrix.data[i].push(if i == j { 1.0 } else { 0.0 });
            }
        }
        aug_matrix.cols *= 2;

        // Apply Gaussian elimination
        if gaussian_elimination(&mut aug_matrix).is_err() {
            return Err("Matrix is singular or nearly singular, cannot be inverted");
        }

        // Transform to echelon form and normalize to get the identity matrix on the left
        for i in 0..matrix.rows {
            let divisor = aug_matrix.data[i][i];
            for j in 0..aug_matrix.cols {
                aug_matrix.data[i][j] /= divisor;
            }

            for k in 0..matrix.rows {
                if k != i {
                    let factor = aug_matrix.data[k][i];
                    for j in 0..aug_matrix.cols {
                        aug_matrix.data[k][j] -= factor * aug_matrix.data[i][j];
                    }
                }
            }
        }

        // Extract the inverse matrix from the augmented matrix
        let inverse_data = aug_matrix.data.iter().map(|row| row[matrix.cols..].to_vec()).collect();
        Ok(Matrix {
            data: inverse_data,
            rows: matrix.rows,
            cols: matrix.cols,
        })
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for row in &self.data {
            for (j, &val) in row.iter().enumerate() {
                if j == 0 {
                    write!(f, "|{:.2}", val)?;
                } else {
                    write!(f, " {:.2}", val)?;
                }
            }
            writeln!(f, "|")?;
        }
        Ok(())
    }
}

impl fmt::Display for Vector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, &val) in self.data.iter().enumerate() {
            if i < self.data.len() - 1 {
                write!(f, "{:.2}, ", val)?;
            } else {
                write!(f, "{:.2}", val)?;
            }
        }
        write!(f, "]")
    }
}

fn main() {

    let mut matrix_a = Matrix {
        data: vec![vec![2.0, 1.0], vec![1.0, -1.0]],
        rows: 2,
        cols: 2,
    };

    // Define the vector b
    let vector_b = vec![3.0, 0.0];

    // Solve Ax = b
    match numerical_methods::solve(&matrix_a, vector_b) {
        Ok(solution) => println!("Solution: {:?}", solution),
        Err(e) => println!("Error: {}", e),
    }

    matrix_a = Matrix {
        data: vec![vec![2.0, 1.0], vec![1.0, -1.0]],
        rows: 2,
        cols: 2,
    };

    // Invert A
    match numerical_methods::invert(&matrix_a) {
        Ok(inverse) => {
            println!("Inverse of A:");
            inverse.print(); // Assuming the print method is correctly implemented
        },
        Err(e) => println!("Error: {}", e),
    }
    let matrix_path = Path::new("matrix.txt");
    let vector_path = Path::new("vec.txt");

    // Read the matrix and vector from files
    let matrix = Matrix::from_file(matrix_path).unwrap();

    let vector = Vector::from_file(vector_path).unwrap();

    // Ensure the vector data is suitable for solving Ax = b
    if vector.len != matrix.rows {
        eprintln!("Error");
    }

    // Solve Ax = b using the Gaussian elimination method
    match numerical_methods::solve(&matrix, vector.data) {
        Ok(solution) => println!("Solution: {}", Vector { data: solution.clone(), len: solution.len() }),
        Err(e) => println!("Error: {}", e),
    };

// Similar adjustment can be made for printing the inverse matrix
    match numerical_methods::invert(&matrix_a) {
        Ok(inverse) => {
            println!("Inverse of A:\n{}", inverse);
        },
        Err(e) => println!("Error: {}", e),
    };
}
