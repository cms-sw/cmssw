/********************************
This is a simple matrix class, with convenient manipulation methods.
Bruce Knuteson 2003
********************************/


#include <vector>
#include <map>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <iterator>

#ifndef __matrix__
#define __matrix__

class matrix
{
public:

  typedef std::vector<double> row_t;
  typedef std::vector<row_t> data_t;
  typedef data_t::size_type size_type;

  /***  Constructors  ***/

  /// Construct by specifying number of rows (n) and columns (m)
  explicit matrix(size_type n=1, size_type m=1);

  /// Construct by providing a (rectangular) vector of vectors
  matrix(data_t const& _data);

  /// Assignment operator
  matrix & operator=(matrix rhs);

  /***   Accessors   ***/

  /// Return the number of rows
  size_type nrows() const;

  /// Return the number of columns
  size_type ncols() const;

  /// Return the <row>^th row
  row_t & operator[](size_type row);

  /// Return the <row>^th row
  const row_t & operator[](size_type row) const;

  /***   Modifiers   ***/

  /// Resize the matrix to n x m
  void resize(size_type n, size_type m);

  /// Delete the m^th column
  void deletecolumn(size_type m);

  /// Delete the n^th row
  void deleterow(size_type n);

  /***    Methods    ***/

  /// Dump this object to an output stream
  void print(std::ostream & fout = std::cout);

  /// Return this matrix raised to the n^th power
  matrix power(double n) const;

  /// Return the inverse of this matrix, together with its determinant
  matrix safeInverse(double & determinant);

  /// Return the eigenvalues of this matrix
  row_t getEigenvalues();

  /// Return the determinant of this matrix
  double det();

  /// Convert this matrix to a (rectangular) vector of vectors
  data_t toSTLVector();

private:

  /// The elements of the matrix
  data_t data;
};


/// Convenient routine from Numerical Recipes
void jacobi(matrix & a, matrix::size_type n, std::vector<double> & d, matrix & v);

// A Utilities routine to allow easy printing of vectors 
template<class T> void print(std::vector<T> vec, std::ostream & fout = std::cout);

template<class T>
void print(std::vector<T> vec, std::ostream & fout)
{
  std::copy( vec.begin(), vec.end(), std::ostream_iterator<T>(fout," "));
  fout << '\n';
}

// A Utilities routine to allow easy reading of vectors 
template<class T> void read(std::vector<T> & vec, std::istream & fin = std::cin, int n=-1);

template<class T>
void read(std::vector<T> & vec, std::istream & fin, int n)
{
  if(n<0)
    fin >> n;
  vec = std::vector<T>(n);
  T blah;
  for(size_t i=0; i<vec.size(); i++)
    {
      fin >> blah;
      vec[i] = blah;
    }
  return;
}

#endif
