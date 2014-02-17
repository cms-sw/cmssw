//
// $Id: matutil.cc,v 1.1 2011/05/26 09:47:00 mseidel Exp $
//
// File: src/matutil.cc
// Purpose: Define matrix types for the hitfit package, and supply a few
//          additional operations.
// Created: Jul, 2000, sss, based on run 1 mass analysis code.
//
// CMSSW File      : src/matutil.cc
// Original Author : Scott Stuart Snyder <snyder@bnl.gov> for D0
// Imported to CMSSW by Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>
//

/**
    @file matutil.cc

    @brief Define matrix types for the HitFit package, and supply a few
    additional operations.  See the documentation for the header file
    matutil.h for details.

    @author Scott Stuart Snyder <snyder@bnl.gov>

    @par Creation date:
    July 2000.

    @par Modification History:
    Apr 2009: Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>:
    Imported to CMSSW.<br>
    Nov 2009: Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>:
    Added doxygen tags for automatic generation of documentation.

    @par Terms of Usage:
    With consent for the original author (Scott Snyder).

 */


#include "TopQuarkAnalysis/TopHitFit/interface/matutil.h"
#include <cassert>


namespace hitfit {


Row_Vector::Row_Vector (int cols)
//
// Purpose: Constructor.
//          Does not initialize the vector.
//
// Inputs:
//   cols -        The length of the vector.
//
  : Matrix (1, cols)
{
}


Row_Vector::Row_Vector (int cols, int /*init*/)
//
// Purpose: Constructor.
//          Initializes the vector to 0.
//
// Inputs:
//   cols -        The length of the vector.
//   init -        Dummy.  Should be 0.
//
  : Matrix (1, cols, 0)
{
}


Row_Vector::Row_Vector (const Matrix& m)
//
// Purpose: Copy constructor.
//          Raises an assertion if M does not have exactly one row.
//
// Inputs:
//   m -           The matrix to copy.
//                 Must have exactly one row.
//
  : Matrix (m)
{
  assert (m.num_row() == 1);
}


const double& Row_Vector::operator() (int col) const
//
// Purpose: Element access.
//
// Inputs:
//   col -         The column to access.  Indexing starts with 1.
//
// Returns:
//   Const reference to the selected element.
//
{
  return HepMatrix::operator() (1, col);
}


double& Row_Vector::operator() (int col)
//
// Purpose: Element access.
//
// Inputs:
//   col -         The column to access.  Indexing starts with 1.
//
// Returns:
//   Reference to the selected element.
//
{
  return HepMatrix::operator() (1, col);
}


const double& Row_Vector::operator() (int row, int col) const
//
// Purpose: Element access.
//
// Inputs:
//   row -         The row to access.  Indexing starts with 1.
//   col -         The column to access.  Indexing starts with 1.
//
// Returns:
//   Const reference to the selected element.
//
{
  return HepMatrix::operator() (row, col);
}


double& Row_Vector::operator() (int row, int col)
//
// Purpose: Element access.
//
// Inputs:
//   row -         The row to access.  Indexing starts with 1.
//   col -         The column to access.  Indexing starts with 1.
//
// Returns:
//   Reference to the selected element.
//
{
  return HepMatrix::operator() (row, col);
}


Row_Vector& Row_Vector::operator= (const Matrix& m)
//
// Purpose: Assignment operator.
//          Raises an assertion if M does not have exactly one row.
//
// Inputs:
//   m -           The matrix to copy.
//                 Must have exactly one row.
//
// Returns:
//   This object.
//
{
  assert (m.num_row() == 1);
  *((Matrix*)this) = m;
  return *this;
}


void clear (CLHEP::HepGenMatrix& m)
//
// Purpose: Reset the matrix M to zero.
//
// Inputs:
//   m -           The matrix to reset.
//
{
  int nrow = m.num_row();
  int ncol = m.num_col();
  for (int i=1; i <= nrow; i++)
    for (int j=1; j <= ncol; j++)
      m(i, j) = 0;
}


double scalar (const CLHEP::HepGenMatrix& m)
//
// Purpose: Return the 1x1 matrix M as a scalar.
//          Raise an assertion if M is not 1x1.
//
// Inputs:
//   m -           The matrix to convert.
//                 Must be 1x1.
//
// Returns:
//   m(1,1)
//
{
  assert (m.num_row() == 1 && m.num_col() == 1);
  return m (1, 1);
}


} // namespace hitfit
