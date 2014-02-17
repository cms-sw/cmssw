//
// $Id: Base_Constrainer.cc,v 1.1 2011/05/26 09:46:59 mseidel Exp $
//
// File: src/Base_Constrainer.cc
// Purpose: Abstract base for the chisq fitter classes.
//          This allows for different algorithms to be used.
// Created: Jul, 2000, sss, based on run 1 mass analysis code.
//
// CMSSW File      : src/Base_Constrainer.cc
// Original Author : Scott Stuart Snyder <snyder@bnl.gov> for D0
// Imported to CMSSW by Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>
//

/**

    @file Base_Constrainer.cc

    @brief Abstract base classes for the \f$\chi^{2}\f$
    fitter classes.  Includes helper function(s).
    See the documentation for header file Base_Constrainer.h for details.

    @par Creation date:
    July 2000.

    @author
    Scott Stuart Snyder <snyder@bnl.gov>.

    @par Modification History:
    Apr 2009: Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>:
    Imported to CMSSW.<br>
    Oct 2009: Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>:
    Added Doxygen tags for automatic generation of documentation.

    @par Terms of Usage:
    With consent from the original author (Scott Snyder).

 */


#include "TopQuarkAnalysis/TopHitFit/interface/Base_Constrainer.h"
#include "TopQuarkAnalysis/TopHitFit/interface/matutil.h"
#include "TopQuarkAnalysis/TopHitFit/interface/Defaults.h"
#include <iostream>
#include <cmath>
#include <cstdlib>

using std::abort;
using std::abs;
using std::cout;
using std::ostream;


//*************************************************************************
// Helper function for doing gradient testing.
//


namespace {

/**
   @brief Test if <i>a</i> is significantly different than <i>b</i>.
   <i>c</i> sets the scale for the comparison, <i>eps</i> gives by
   how much they may differ.

   @param a First number to be compared.
   @param b Second number to be compared.
   @param c Scale of the comparison.
   @param eps How much the two numbers may be different.
   @par Return:
   Returns true if one of the two numbers is significantly larger than
   the other one, otherwise false.
 */
bool test_different (double a, double b, double c, double eps)
//
// Purpose: Test if A is significantly different than B.
//          C sets the scale for the comparison; EPS gives
//          by how much they may differ.
//
{
  double scale = eps * (abs (a) + abs (b) + abs (c));
  if (abs (a) != 0 && abs (b) / abs (a) < 0.1 && abs (a) > scale)
    scale = abs (a) * .5;
  if (scale == 0) return false;
  if (scale < eps) scale = eps;
  return abs (a - b) > scale;
}


} // unnamed namespace


//*************************************************************************


namespace hitfit {


//*************************************************************************


Base_Constrainer_Args::Base_Constrainer_Args (const Defaults& defs)
//
// Purpose: Constructor.
//
// Inputs:
//   defs -        The Defaults instance from which to initialize.
//
  : _test_gradient (defs.get_bool ("test_gradient")),
    _test_step (defs.get_float ("test_step")),
    _test_eps (defs.get_float ("test_eps"))
{
}


bool Base_Constrainer_Args::test_gradient () const
//
// Purpose: Return the test_gradient parameter.
//          See the header for documentation.
//
{
  return _test_gradient;
}


double Base_Constrainer_Args::test_step () const
//
// Purpose: Return the test_step parameter.
//          See the header for documentation.
//
{
  return _test_step;
}


double Base_Constrainer_Args::test_eps () const
//
// Purpose: Return the test_eps parameter.
//          See the header for documentation.
//
{
  return _test_eps;
}


//*************************************************************************


Constraint_Calculator::Constraint_Calculator (int nconstraints)
//
// Purpose: Constructor.
//
// Inputs:
//   nconstraints- The number of constraint functions.
//
  : _nconstraints (nconstraints)
{
}


int Constraint_Calculator::nconstraints () const
//
// Purpose: Return the number of constraint functions.
//
// Returns:
//   The number of constraint functions.
//
{
  return _nconstraints;
}


//*************************************************************************


Base_Constrainer::Base_Constrainer (const Base_Constrainer_Args& args)
//
// Purpose: Constructor.
//
// Inputs:
//   args -        The parameter settings for this instance.
//
  : _args (args)
{
}


std::ostream& Base_Constrainer::print (std::ostream& s) const
//
// Purpose: Print our state.
//
// Inputs:
//   s -           The stream to which to write.
//
// Returns:
//   The stream S.
//
{
  s << "Base_Constrainer parameters:\n";
  s << " test_gradient: " << _args.test_gradient()
    << " test_step: " << _args.test_step()
    << " test_eps: " << _args.test_eps() << "\n";
  return s;
}


/**
    @brief Output stream operator, print the content of this Base_Constrainer
    to an output stream.

    @param s The output stream to which to write.

    @param f The instance of Base_Constrainer to be printed.

*/
std::ostream& operator<< (std::ostream& s, const Base_Constrainer& f)
//
// Purpose: Print our state.
//
// Inputs:
//   s -           The stream to which to write.
//   f -           The instance to dump.
//
// Returns:
//   The stream S.
//
{
  return f.print (s);
}


bool Base_Constrainer::call_constraint_fcn (Constraint_Calculator&
                                            constraint_calculator,
                                            const Column_Vector& x,
                                            const Column_Vector& y,
                                            Row_Vector& F,
                                            Matrix& Bx,
                                            Matrix& By) const
//
// Purpose: Call the constraint function for the point x, y.
//          Return F, Bx, By, and a flag saying if the
//          point is acceptable.
//
//          If test_gradient is on, we verify the gradients returned
//          by also computing them numerically.
//
// Inputs:
//   constraints - The user-supplied object to evaluate the constraints.
//   x(Nw) -       Vector of well-measured quantities where we evaluate
//                 the constraints.
//   y(Np) -       Vector of poorly-measured quantities where we evaluate
//                 the constraints.
//
// Outputs:
//   F(Nc) -       The results of the constraint functions.
//   Bx(Nw,Nc) -   Gradients of F with respect to x.
//   By(Np,Nc) -   Gradients of F with respect to y.
//
// Returns:
//   True if the point is accepted, false if it was rejected.
{
  // Call the user's function.
  bool val = constraint_calculator.eval (x, y, F, Bx, By);

  // If we're not doing gradients numerically, we're done.
  if (!_args.test_gradient())
    return val;

  // Bail if the point was rejected.
  if (!val)
    return false;

  int Nw = x.num_row();
  int Np = y.num_row();
  int Nc = F.num_col();

  // Numerically check Bx.
  for (int i=1; i<=Nc; i++) {
    // Step a little along variable I.
    Column_Vector step_x (Nw, 0);
    step_x(i) = _args.test_step();
    Column_Vector new_x = x + step_x;

    // Evaluate the constraints at the new point.
    Matrix new_Bx (Nw, Nc);
    Matrix new_By (Np, Nc);
    Row_Vector new_F (Nc);
    if (! constraint_calculator.eval (new_x, y, new_F, new_Bx, new_By))
      return false;

    // Calculate what we expect the constraints to be at this point,
    // given the user's gradients.
    Row_Vector test_F = F + step_x.T() * Bx;

    // Check the results.
    for (int j=1; j<=Nc; j++) {
      if (test_different (test_F(j), new_F(j), F(j), _args.test_eps())) {
        cout << "bad gradient x " << i << " " << j << "\n";
        cout << x;
        cout << y;
        cout << new_x;
        cout << F;
        cout << new_F;
        cout << Bx;
        cout << (test_F - new_F);
        abort ();
      }
    }
  }

  // Numerically check By.
  for (int i=1; i<=Np; i++) {
    // Step a little along variable I.
    Column_Vector step_y (Np, 0);
    step_y(i) = _args.test_step();
    Column_Vector new_y = y + step_y;

    // Evaluate the constraints at the new point.
    Matrix new_Bx (Nw, Nc);
    Matrix new_By (Np, Nc);
    Row_Vector new_F (Nc);
    if (! constraint_calculator.eval (x, new_y, new_F, new_Bx, new_By))
      return false;

    // Calculate what we expect the constraints to be at this point,
    // given the user's gradients.
    Row_Vector test_F = F + step_y.T() * By;

    // Check the results.
    for (int j=1; j<=Nc; j++) {
      if (test_different (test_F(j), new_F(j), F(j), _args.test_eps())) {
        cout << "bad gradient y " << i << " " << j << "\n";
        cout << x;
        cout << y;
        cout << new_y;
        cout << F;
        cout << new_F;
        cout << Bx;
        cout << (test_F - new_F);
        abort ();
      }
    }
  }

  // Done!
  return true;
}


} // namespace hitfit
