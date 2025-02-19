//
// $Id: Chisq_Constrainer.cc,v 1.2 2011/10/13 09:49:48 snaumann Exp $
//
// File: src/Chisq_Constrainer.cc
// Purpose: Minimize a chisq subject to a set of constraints.
//          Based on the SQUAW algorithm.
// Created: Jul, 2000, sss, based on run 1 mass analysis code.
//
// CMSSW File      : src/Chisq_Constrainer.cc
// Original Author : Scott Stuart Snyder <snyder@bnl.gov> for D0
// Imported to CMSSW by Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>
//

/**
    @file Chisq_Constrainer.cc

    @brief Minimize a \f$\chi^{2}\f$ subject to a set of constraints,
    based on the SQUAW algorithm.  See the documentation for header
    file Chisq_Constrainer.h for details.

    @par Creation date.
    July 2000.

    @author
    Scott Stuart Snyder <snyder@bnl.gov>.

    @par Modification History
    Apr 2009: Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>:
    Imported to CMSSW.<br>
    Oct 2009: Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>:
    Added Doxygen tags for automatic generation of documentation.

    @par Terms of Usage.
    With consent from the original author (Scott Snyder).

*/


#include "TopQuarkAnalysis/TopHitFit/interface/Chisq_Constrainer.h"
#include "TopQuarkAnalysis/TopHitFit/interface/Defaults.h"
#include <cmath>
#include <cassert>
#include <iostream>
#include <iomanip>

using std::abs;
using std::cout;
using std::fixed;
using std::ios_base;
using std::ostream;
using std::resetiosflags;
using std::setiosflags;
using std::sqrt;


//*************************************************************************


namespace hitfit {


Chisq_Constrainer_Args::Chisq_Constrainer_Args (const Defaults& defs)
//
// Purpose: Constructor.
//
// Inputs:
//   defs -        The Defaults instance from which to initialize.
//
  : _base_constrainer_args (defs)
{
  _use_G = defs.get_bool ("use_G");
  _printfit = defs.get_bool ("printfit");
  _constraint_sum_eps = defs.get_float ("constraint_sum_eps");
  _chisq_diff_eps = defs.get_float ("chisq_diff_eps");
  _maxit = defs.get_int ("maxit");
  _max_cut = defs.get_int ("maxcut");
  _cutsize = defs.get_float ("cutsize");
  _min_tot_cutsize = defs.get_float ("min_tot_cutsize");
  _chisq_test_eps = defs.get_float ("chisq_test_eps");
}


bool Chisq_Constrainer_Args::printfit () const
//
// Purpose: Return the printfit parameter.
//          See the header for documentation.
//
{
  return _printfit;
}


bool Chisq_Constrainer_Args::use_G () const
//
// Purpose: Return the use_G parameter.
//          See the header for documentation.
//
{
  return _use_G;
}


double Chisq_Constrainer_Args::constraint_sum_eps () const
//
// Purpose: Return the constraint_sum_eps parameter.
//          See the header for documentation.
//
{
  return _constraint_sum_eps;
}


double Chisq_Constrainer_Args::chisq_diff_eps () const
//
// Purpose: Return the chisq_diff_eps parameter.
//          See the header for documentation.
//
{
  return _chisq_diff_eps;
}


unsigned  Chisq_Constrainer_Args::maxit () const
//
// Purpose: Return the maxit parameter.
//          See the header for documentation.
//
{
  return _maxit;
}


unsigned  Chisq_Constrainer_Args::max_cut () const
//
// Purpose: Return the max_cut parameter.
//          See the header for documentation.
//
{
  return _max_cut;
}


double Chisq_Constrainer_Args::cutsize () const
//
// Purpose: Return the cutsize parameter.
//          See the header for documentation.
//
{
  return _cutsize;
}


double Chisq_Constrainer_Args::min_tot_cutsize () const
//
// Purpose: Return the min_tot_cutsize parameter.
//          See the header for documentation.
//
{
  return _min_tot_cutsize;
}


double Chisq_Constrainer_Args::chisq_test_eps () const
//
// Purpose: Return the chisq_test_eps parameter.
//          See the header for documentation.
//
{
  return _chisq_test_eps;
}


const Base_Constrainer_Args&
Chisq_Constrainer_Args::base_constrainer_args () const
//
// Purpose: Return the contained subobject parameters.
//          See the header for documentation.
//
{
  return _base_constrainer_args;
}


} // namespace hitfit


//*************************************************************************


namespace {


using namespace hitfit;


/**
     Solve the linear system<br>
     \f$\left( \begin{array}{rr} -H & B \cdot t \\ B & Y \end{array}\right)
     \left(\begin{array}{r} alpha \\ d \end{array} \right) =
     \left(\begin{array}{r}     r \\ 0 \end{array} \right)\f$ for <i>alpha</i> and <i>d</i>.

     Also returns the inverse matrices<br>
     \f$
     \left(\begin{array}{rr} W & V \cdot t \\ V & U\end{array}\right) =
     \left(\begin{array}{rr}-H & B \cdot t \\ B & Y\end{array}\right)^{-1}
     \f$

     @par Return:
     <b>true</b> if successful.<br>
     <b>false</b> if fail.
*/
bool solve_linear_system (const Matrix& H,
                          const Diagonal_Matrix& Y,
                          const Matrix& By,
                          const Row_Vector& r,
                          Column_Vector& alpha,
                          Column_Vector& d,
                          Matrix& W,
                          Matrix& U,
                          Matrix& V)
//
// Purpose: Solve the system
//
//   [ -H  B.t ]   [ alpha ]     [ r ]
//   [         ] * [       ]  =  [   ]
//   [  B  Y   ]   [   d   ]     [ 0 ]
// 
//  for alpha and d.
// 
//  Also returns the inverse matrices:
// 
//   [ W  V.t ]     [ -H  B.t ]
//   [        ]  =  [         ] ^ -1
//   [ V  U   ]     [  B  Y   ]
// 
//  Returns true if successful, false if not.
//
{
  int nconstraints = H.num_row();
  int nbadvars = Y.num_row();

  // Form the matrix on the LHS from H, By, and Y.
  Matrix A (nconstraints+nbadvars, nconstraints+nbadvars);
  A.sub (1, 1, -H);
  if (nbadvars > 0) {
    A.sub (nconstraints+1, nconstraints+1, Y);
    A.sub (1, nconstraints+1, By.T());
    A.sub (nconstraints+1, 1, By);
  }

  // Form the RHS vector from r.
  Column_Vector yy(nconstraints + nbadvars, 0);
  yy.sub (1, r.T());

  // Invert the matrix.
  // Try to handle singularities correctly.
  Matrix Ai;
  int ierr = 0;
  do {
    Ai = A.inverse (ierr);
    if (ierr) {
      int allzero = 0;
      for (int i=1; i<=nconstraints; i++) {
    allzero = 1;
    for (int j=1; j<=nconstraints; j++) {
      if (A(i,j) != 0) {
        allzero = 0;
        break;
      }
    }
    if (allzero) {
      A(i,i) = 1;
      break;
    }
      }
      if (!allzero) return false;
    }
  } while (ierr != 0);

  // Solve the system of equations.
  Column_Vector xx = Ai * yy;

  // Extract the needed pieces from the inverted matrix
  // and the solution vector.
  W = Ai.sub (1, nconstraints, 1, nconstraints);
  if (nbadvars > 0) {
    U = Ai.sub (nconstraints+1, nconstraints+nbadvars,
                nconstraints+1, nconstraints+nbadvars);
    V = Ai.sub (nconstraints+1, nconstraints+nbadvars, 1, nconstraints);
    d = xx.sub (nconstraints+1, nconstraints+nbadvars);
  }

  alpha = xx.sub (1, nconstraints);

  return true;
}


} // unnamed namespace


namespace hitfit {


//*************************************************************************


Chisq_Constrainer::Chisq_Constrainer (const Chisq_Constrainer_Args& args)
//
// Purpose: Constructor.
//
// Inputs:
//   args -        The parameter settings for this instance.
//
  : Base_Constrainer (args.base_constrainer_args()),
    _args (args)
{
}


double Chisq_Constrainer::fit (Constraint_Calculator& constraint_calculator,
                               const Column_Vector& xm,
                               Column_Vector& x,
                               const Column_Vector& ym,
                               Column_Vector& y,
                               const Matrix& G_i,
                               const Diagonal_Matrix& Y,
                               Column_Vector& pullx,
                               Column_Vector& pully,
                               Matrix& Q,
                               Matrix& R,
                               Matrix& S)
//
// Purpose: Do a constrained fit.
//
// Call the number of well-measured variables Nw, the number of
// poorly-measured variables Np, and the number of constraints Nc.
//
// Inputs:
//   constraint_calculator - The object that will be used to evaluate
//                   the constraints.
//   xm(Nw)      - The measured values of the well-measured variables.
//   ym(Np)      - The measured values of the poorly-measured variables.
//   x(Nw)       - The starting values for the well-measured variables.
//   y(Np)       - The starting values for the poorly-measured variables.
//   G_i(Nw,Nw)  - The error matrix for the well-measured variables.
//   Y(Np,Np)    - The inverse error matrix for the poorly-measured variables.
//
// Outputs:
//   x(Nw)       - The fit values of the well-measured variables.
//   y(Np)       - The fit values of the poorly-measured variables.
//   pullx(Nw)   - The pull quantities for the well-measured variables.
//   pully(Nw)   - The pull quantities for the poorly-measured variables.
//   Q(Nw,Nw)    - The final error matrix for the well-measured variables.
//   R(Np,Np)    - The final error matrix for the poorly-measured variables.
//   S(Nw,Np)    - The final cross error matrix for the two sets of variables.
//
// Returns:
//   The minimum chisq satisfying the constraints.
//   Returns a value < 0 if the fit failed to converge.
//
{
  // Check that the various matrices we've been passed have consistent
  // dimensionalities.
  int nvars = x.num_row();
  assert (nvars == G_i.num_col());
  assert (nvars == xm.num_row());

  int nbadvars = y.num_row();
  assert (nbadvars == Y.num_col());
  assert (nbadvars == ym.num_row());

  // If we're going to check the chisq calculation by explicitly using G,
  // calculate it now from its inverse G_i.
  Matrix G (nvars, nvars);
  if (_args.use_G()) {
    int ierr = 0;
    G = G_i.inverse (ierr);
    assert (!ierr);
  }

  int nconstraints = constraint_calculator.nconstraints ();

  // Results of the constraint evaluation function.
  Row_Vector F (nconstraints);             // Constraint vector.
  Matrix Bx (nvars, nconstraints);         // Gradients wrt x
  Matrix By (nbadvars, nconstraints);      // Gradients wrt y

  // (2) Evaluate the constraints at the starting point.
  // If the starting point is rejected as invalid,
  // give up and return an error.
  if (! call_constraint_fcn (constraint_calculator, x, y, F, Bx, By)) {
      //    cout << "Bad initial values!";
      //    return -1000;
      return -999.0;
  }

  // (3) Initialize variables for the fitting loop.
  double constraint_sum_last = -1000;
  double chisq_last = -1000;
  bool near_convergence = false;
  double last_step_cutsize = 1;

  unsigned nit = 0;

  // Initialize the displacement vectors c and d.
  Column_Vector c = x - xm;
  Column_Vector d = y - ym;

  Matrix E (nvars, nconstraints);
  Matrix W (nconstraints, nconstraints);
  Matrix U (nbadvars, nbadvars);
  Matrix V (nbadvars, nconstraints);

  // (4) Fitting loop:
  do {
    // (5) Calculate E, H, and r.
    E = G_i * Bx;
    Matrix H = E.T() * Bx;
    Row_Vector r = c.T() * Bx + d.T() * By - F;

    // (6) Solve the linearized system for the new values
    // of the Lagrange multipliers
    // $\alpha$ and the new value for the displacements d.
    Column_Vector alpha (nvars);
    Column_Vector d1 (nbadvars);
    if (!solve_linear_system (H, Y, By, r,
                              alpha, d1, W, U, V)) {
        ///      cout << "singular matrix!";
        //      return -1000;
        return -998.0;
    }

    // (7) Compute the new values for the displacements c and the chisq.
    Column_Vector c1 = -E * alpha;
    double chisq =  - scalar (r * alpha);

    double psi_cut = 0;

    // (8) Find where this step is going to be taking us.
    x = c1 + xm;
    y = d1 + ym;

    // (9) Set up for cutting this step, should we have to.
    Matrix save_By = By;
    Row_Vector save_negF = - F;
    double this_step_cutsize = 1;
    double constraint_sum = -1;
    unsigned ncut = 0;

    // (10) Evaluate the constraints at the new point.
    // If the point is rejected, we have to try to cut the step.
    // We accept the step if:
    //  The constraint sum is below the convergence threshold
    //    constraint_sum_eps, or
    //  This is the first iteration, or
    //  The constraint sum has decreased since the last iteration.
    // Otherwise, the constraints have gotten worse, and we
    // try to cut the step.
    while (! call_constraint_fcn (constraint_calculator, x, y, F, Bx, By) ||
       ((constraint_sum = norm_infinity (F))
              > _args.constraint_sum_eps() &&
        nit > 0 &&
        constraint_sum > constraint_sum_last))
    {

      // Doing step cutting...
      if (nit > 0 && _args.printfit() && ncut == 0) {
    cout << "(" << chisq << " " << chisq_last << ") ";
      }

      // (10a) If this is the first time we've tried to cut this step,
      // test to see if the chisq is stationary.  If it hasn't changed
      // since the last iteration, try a directed step.
      if (ncut == 0 &&
      abs (chisq - chisq_last) < _args.chisq_diff_eps()) {

    // Trying a directed step now.
    // Try to make the smallest step which satisfies the
    // (linearized) constraints.
    if (_args.printfit())
      cout << " directed step ";

    // (10a.i) Solve the linearized system for $\beta$ and
    // the y-displacement vector $\delta$.
    Column_Vector beta (nconstraints);
    Column_Vector delta (nbadvars);
    solve_linear_system (H, Y, save_By, save_negF,
                 beta, delta, W, U, V);

    // (10a.ii) Get the x-displacement vector $\gamma$.
    Column_Vector gamma = -E * beta;

    // (10a.iii) Find the destination of the directed step.
    x = c + xm + gamma;
    y = d + ym + delta;

    // (10a.iv) Accept this point if it's not rejected by the constraint
    // function, and the constraints improve.
    if (call_constraint_fcn (constraint_calculator, x, y, F, Bx, By) &&
        (constraint_sum = norm_infinity (F)) > 0 &&
        (constraint_sum < constraint_sum_last)) {

      // Accept this step.  Calculate the chisq and new displacement
      // vectors.
      chisq = chisq_last - scalar ((-save_negF + r*2) * beta);
      c1 = x - xm;
      d1 = y - ym;

      // Exit from step cutting loop.
      break;
    }
      }

      // If this is the first time we're cutting the step,
      // initialize $\psi$.
      if (ncut == 0)
    psi_cut = scalar ((save_negF - r) * alpha);

      // (10b) Give up if we've tried to cut this step too many times.
      if (++ncut > _args.max_cut()) {
          //    cout << " Too many cut steps ";
          //    return -1000;
          return -997.0;
      }

      // (10c) Set up the size by which we're going to cut this step.
      // Normally, this is cutsize.  But if this is the first time we're
      // cutting this step and the last step was also cut, set the cut
      // size to twice the final cut size from the last step (provided
      // that it is less than cutsize).
      double this_cutsize = _args.cutsize();
      if (ncut == 1 && last_step_cutsize < 1) {
    this_cutsize = 2 * last_step_cutsize;
    if (this_cutsize > _args.cutsize())
      this_cutsize = _args.cutsize();
      }

      // (10d) Keep track of the total amount by which we've cut this step.
      this_step_cutsize *= this_cutsize;

      // If it falls below min_tot_cutsize, give up.
      if (this_step_cutsize < _args.min_tot_cutsize()) {
          //    cout << "Cut size underflow ";
          //    return -1000;
          return -996.0;
      }

      // (10e) Cut the step: calculate the new displacement vectors.
      double cutleft = 1 - this_cutsize;
      c1 = c1 * this_cutsize + c * cutleft;
      d1 = d1 * this_cutsize + d * cutleft;

      // (10f) Calculate the new chisq.
      if (chisq_last >= 0) {
    chisq = this_cutsize*this_cutsize * chisq +
            cutleft*cutleft * chisq_last +
               2*this_cutsize*cutleft * psi_cut;
    psi_cut = this_cutsize * psi_cut + cutleft * chisq_last;
      }
      else
    chisq = chisq_last;

      // Log what we've done.
      if (_args.printfit()) {
        cout << constraint_sum << " cut " << ncut << " size "
             << setiosflags (ios_base::scientific)
             << this_cutsize << " tot size " << this_step_cutsize
             << resetiosflags (ios_base::scientific)
             << " " << chisq << "\n";
      }

      // Find the new step destination.
      x = c1 + xm;
      y = d1 + ym;

      // Now, go and test the step again for acceptability.
    }

    // (11) At this point, we have an acceptable step.
    // Shuffle things around to prepare for the next step.
    last_step_cutsize = this_step_cutsize;

    // If requested, calculate the chisq using G to test for
    // possible loss of precision.
    double chisq_b = 0;
    if (_args.use_G()) {
      chisq_b = scalar (c1.T() * G * c1) + scalar (d1.T() * Y * d1);
      if (chisq >= 0 &&
      abs ((chisq - chisq_b) / chisq) > _args.chisq_test_eps()) {
    cout << chisq << " " << chisq_b
         << "lost precision?\n";
    abort ();
      }
    }

    // Log what we're doing.
    if (_args.printfit()) {
      cout << chisq << " ";
      if (_args.use_G())
    cout << chisq_b << " ";
    }

    double z2 = abs (chisq - chisq_last);

    if (_args.printfit()) {
      cout << constraint_sum << " " << z2 << "\n";
    }

    c = c1;
    d = d1;
    chisq_last = chisq;
    constraint_sum_last = constraint_sum;

    // (12) Test for convergence.  The conditions must be satisfied
    // for two iterations in a row.
    if (chisq >= 0 && constraint_sum < _args.constraint_sum_eps() &&
    z2 < _args.chisq_diff_eps())
    {
      if (near_convergence) break;  // Converged!  Exit loop.
      near_convergence = true;
    }
    else
      near_convergence = false;

    // (13) Give up if we've done this too many times.
    if (++nit > _args.maxit()) {
        //      cout << "too many iterations";
        //      return -1000;
        return -995.0;
    }

  } while (1);

  // (15) Ok, we have a successful fit!


  // Calculate the error matrices.
  Q = E * W * E.T();
  S = - E * V.T();
  R = U;

  // And the vectors of pull functions.
  pullx = Column_Vector (nvars);
  for (int i=1; i<=nvars; i++) {
    double a = Q(i,i);
    if (a < 0)
      pullx(i) = c(i) / sqrt (-a);
    else {
      pullx(i) = 0;
      //      cout << " bad pull fcn for var " << i << " (" << a << ") ";
    }
  }

  pully = Column_Vector (nbadvars);
  for (int i=1; i<=nbadvars; i++) {
    double a = 1 - Y(i,i)*R(i,i);
    if (a > 0)
      pully(i) = d(i) * sqrt (Y(i,i) / a);
    else {
      pully(i) = 0;
      //      cout << " bad pull fcn for badvar " << i << " ";
    }
  }

  // Finish calculation of Q.
  Q = Q + G_i;

  // Return the final chisq.
  return chisq_last;
}


std::ostream& Chisq_Constrainer::print (std:: ostream& s) const
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
  Base_Constrainer::print (s);
  s << " printfit: " << _args.printfit()
    << "  use_G: " << _args.use_G() << "\n";
  s << " constraint_sum_eps: " << _args.constraint_sum_eps()
    << "  chisq_diff_eps: " << _args.chisq_diff_eps()
    << "  chisq_test_eps: " << _args.chisq_test_eps() << "\n";
  s << " maxit: " << _args.maxit()
    << "  max_cut: " << _args.max_cut()
    << "  min_tot_cutsize: " << _args.min_tot_cutsize()
    << "  cutsize: " << _args.cutsize() << "\n";
  return s;
}


} // namespace hitfit
