//
// $Id: matutil.h,v 1.1 2011/05/26 09:46:53 mseidel Exp $
//
// File: hitfit/matutil.h
// Purpose: Define matrix types for the hitfit package, and supply a few
//          additional operations.
// Created: Jul, 2000, sss, based on run 1 mass analysis code.
//
// This file defines the types `Matrix', `Column_Vector', `Row_Vector',
// and `Diagonal_Matrix', to be used in hitfit code.  These are based
// on the corresponding CLHEP matrix classes (except that we need
// to create our own row vector class, since CLHEP doesn't have that
// concept).  We also provide a handful of operations that are missing
// in CLHEP.
//
// CMSSW File      : interface/matutil.h
// Original Author : Scott Stuart Snyder <snyder@bnl.gov> for D0
// Imported to CMSSW by Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>
//


/**
    @file matutil.h

    @brief Define matrix types for the HitFit package, and supply a few
    additional operations.

    This file defines the type <i>Matrix</i>, <i>Column_Vector</i>,
    <i>Row_Vector</i>, and <i>Diagonal_Matrix</i>, to be used in HitFit code.
    These are based on the corresponding CLHEP matrix classes (except that
    HitFit uses its own row vector class, since CLHEP doesn't have that concept).
    Also provided are a handful of operations that are missing in CLHEP.

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

#ifndef HITFIT_MATUTIL_H
#define HITFIT_MATUTIL_H

// We want bounds checking.
#define MATRIX_BOUND_CHECK

//#include "CLHEP/config/CLHEP.h"
#include "CLHEP/Matrix/Matrix.h"
#include "CLHEP/Matrix/Vector.h"
#include "CLHEP/Matrix/DiagMatrix.h"


namespace hitfit {


// We use these CLHEP classes as-is.
typedef CLHEP::HepMatrix Matrix;
typedef CLHEP::HepVector Column_Vector;
typedef CLHEP::HepDiagMatrix Diagonal_Matrix;


// CLHEP doesn't have a row-vector class, so make our own.
// This is only a simple wrapper around Matrix that tries to constrain
// the shape to a row vector.  It will raise an assertion if you try
// to assign to it something that isn't a row vector.
/**
    @class Row_Vector

    @brief Row-vector class.  CLHEP doesn't have a row-vector class,
    so HitFit uses its own. This is only a simple wrapper around Matrix
    that tries to constrain the shape to a row vector.  It will raise
    an assertion if you try to assign to it something that isn't a row
    vector.
 */
class Row_Vector
  : public Matrix
//
// Purpose: Simple Row_Vector wrapper for CLHEP matrix class.
//
{
public:
   // Constructor.  Gives an uninitialized 1 x cols matrix.
  /**
     @brief Constructor, instantiate an unitialized
     \f$1 \times \mathrm{cols}\f$ matrix.

     @param cols The number of columns (the length of the vector).
   */
  explicit Row_Vector (int cols);

   // Constructor.  Gives a 1 x cols matrix, initialized to zero.
   // The INIT argument is just a dummy; give it a value of 0.
  /**
     @brief Constructor, instantiate an unitialized
     \f$1 \times \mathrm{cols}\f$ matrix, and initialized it to zero.

     @param cols The number of columns (the length of the vector).

     @param init A dummy argument, should always be zero.
   */
  explicit Row_Vector (int cols, int init);

  // Copy constructor.  Will raise an assertion if M doesn't have
  // exactly one row.
  /**
     @brief Copy constructor, will raise an assertion if <i>m</i>
     doesn't have exactly one row.

     @param m The matrix to copy, must have exactly one row.
   */
  Row_Vector (const Matrix& m);

  // Element access.
  // ** Note that the indexing starts from (1). **
  /**
     @brief Direct element access, indexing starts from 1.

     @param col The column to access.
   */
  const double & operator() (int col) const;

  /**
     @brief Direct element access, indexing starts from 1.

     @param col The column to access.
   */
  double & operator() (int col);

  // Element access.
  // ** Note that the indexing starts from (1,1). **
  /**
     @brief Direct element access, indexing starts from (1,1).

     @param row The row to access.

     @param col The column to access.
   */
  const double & operator()(int row, int col) const;

  /**
     @brief Direct element access, indexing starts from (1,1).

     @param row The row to access.

     @param col The column to access.
   */
  double & operator()(int row, int col);

  // Assignment.  Will raise an assertion if M doesn't have
  // exactly one row.
  /**
     @brief Assignment operator, will raise an assertion if <i>m</i>
     doesn't have exactly one row.

     @param m The matrix to copy, must have exactly one row.
   */
  Row_Vector& operator= (const Matrix& m);
};


// Additional operations.


// Reset all elements of a matrix to 0.
/**
    @brief Helper function: Reset all elements of a matrix to 0.

    @param m The matrix to reset.
 */
void clear (CLHEP::HepGenMatrix& m);

// Check that M has dimensions 1x1.  If so, return the single element
// as a scalar.  Otherwise, raise an assertion failure.
/**
    @brief Return the \f$1 \times 1\f$ matrix as a scalar.
    Raise an assertion if the matris is not \f$1 \times 1\f$.

    @param m The matrix to convert, must be \f$1 \times 1\f$.
 */
double scalar (const CLHEP::HepGenMatrix& m);


} // namespace hitfit



#endif // not HITFIT_MATUTIL_H

