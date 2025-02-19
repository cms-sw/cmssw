//
//   $Id: Base_Constrainer.h,v 1.1 2011/05/26 09:46:52 mseidel Exp $
//
// File: hitfit/Base_Constrainer.h
// Purpose: Abstract base for the chisq fitter classes.
//          This allows for different algorithms to be used.
// Created: Jul, 2000, sss, based on run 1 mass analysis code.
//
// CMSSW File      : interface/Base_Constrainer.h
// Original Author : Scott Stuart Snyder <snyder@bnl.gov> for D0
// Imported to CMSSW by Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>

/**

    @file Base_Constrainer.h

    @brief Abstract base classes for the \f$\chi^{2}\f$ fitter classes.

    This file contains abstract base classes for the \f$\chi^{2}\f$ fitter
    classes.  By doing so, different fitting algorithm may be used.

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
#ifndef HITFIT_BASE_CONSTRAINER_H
#define HITFIT_BASE_CONSTRAINER_H


#include "TopQuarkAnalysis/TopHitFit/interface/matutil.h"
#include <iosfwd>


namespace hitfit {

class Defaults;


//*************************************************************************


/**
    @class Base_Constrainer_Args

    @brief Hold on to parameters for the Base_Constrainer class.
 */
class Base_Constrainer_Args
//
// Purpose: Hold on to parameters for the Base_Constrainer class.
//
// Parameters:
//   bool test_gradient - If true, check the constraint gradient calculations
//                        by also doing them numerically.
//   float test_step    - When test_gradient is true, the step size to use
//                        for numeric differentiation.
//   float test_eps     - When test_gradient is true, the maximum relative
//                        difference permitted between returned and
//                        numerically calculated gradients.
//
{
public:
  // Constructor.  Initialize from a Defaults object.

  /**
     Instantiate Base_Constrainer_Args from an instance of Defaults object.

     @param defs An instance of Defaults object. The instance must contain the
     variables of type and name:
     - bool <i>test_gradient</i>.
     - double <i>test_step</i>.
     - double <i>test_eps</i>.
   */
  Base_Constrainer_Args (const Defaults& defs);

  // Retrieve parameter values.
  /**
     Return the <i>_test_gradient</i> parameter.
   */
  bool test_gradient () const;

  /**
     Return the <i>_test_step</i> parameter.
   */
  double test_step () const;

  /**
     Return the <i>_test_eps</i> parameter.
   */
  double test_eps () const;


private:

  // Hold on to parameter values.

  /**
     If true, check constraint gradient calculation by also doing
     them numerically.
   */
  bool _test_gradient;

  /**
     When <i>_test_gradient</i> is true, the step size use for numerical
     differentation.
   */
  double _test_step;

  /**
     When <i>_test_gradient</i> is true, the maximum relative difference
     permitted between returned and numerically calculated gradients.
   */
  double _test_eps;

};


//*************************************************************************


/**

   @class Constraint_Calculator

   @brief Abstract base class for evaluating constraints.  Users derive
   from this class and implement the eval() method.

 */
class Constraint_Calculator
//
// Purpose: Abstract base class for evaluating constraints.
//          Derive from this and implement the eval() method.
//
{
public:
  // Constructor, destructor.  Pass in the number of constraints.

  /**
     Constructor.
     @param nconstraints Number of constraint equations.
   */
  Constraint_Calculator (int nconstraints);

  /**
     Destructor.
   */
  virtual ~Constraint_Calculator () {}

  // Get back the number of constraints.
  /**
     Return the number of constraints.
   */
  int nconstraints () const;

  // Evaluate constraints at the point described by X and Y (well-measured
  // and poorly-measured variables, respectively).  The results should
  // be stored in F.  BX and BY should be set to the gradients of F with
  // respect to X and Y, respectively.
  //
  // Return true if the point X, Y is accepted.
  // Return false if it is rejected (i.e., in an unphysical region).
  // The constraints need not be evaluated in that case.

  /**
    @brief Evaluate constraints at the point described by <i>x</i> and
    <i>y</i>
    (well-measured and poorly-measured variables, respectively).  The results
    should be stored in <i>F</i>.  <i>Bx</i> and <i>By</i> should be set to
    the gradients of <i>F</i> with respect to <i>x</i> and <i>y</i>,
    respectively.

    @param x  Column_Vector of well-measured variables.

    @param y  Column_Vector of poorly-measured variables.

    @param F  Row_Vector contains the results of the constraint evaluation.

    @param Bx Gradients of <i>F</i> with respect to <i>x</i>

    @param By Gradients of <i>F</i> with respect to <i>y</i>

    @par Output:
    - <i>F</i>.
    - <i>Bx</i>.
    - <i>By</i>.

    @par Return:
    <b>true</b> if the point <i>(x,y)</i> is accepted.<br>
    <b>false</b> if the point <i>(x,y)</i> is rejected
    (i.e., in an unphysical region).  The constraints need not be
    evaluated in that case.

   */
  virtual bool eval (const Column_Vector& x,
                     const Column_Vector& y,
                     Row_Vector& F,
                     Matrix& Bx,
                     Matrix& By) = 0;


private:
  // The number of constraint functions.
  /**
     Number of constraints functions.
   */
  int _nconstraints;

};


//*************************************************************************


/**
    @class Base_Constrainer
    @brief Base class for \f$\chi^{2}\f$ constrained fitter.
 */
class Base_Constrainer
//
// Purpose: Base class for chisq constrained fitter.
//
{
public:
  // Constructor, destructor.
  // ARGS holds the parameter settings for this instance.

  /**
     Constructor.
     @param args Contains the parameter settings for this instance.
   */
  Base_Constrainer (const Base_Constrainer_Args& args);

  /**
     Destructor.
   */
  virtual ~Base_Constrainer () {}

  // Do the fit.
  // Call the number of well-measured variables Nw, the number of
  // poorly-measured variables Np, and the number of constraints Nc.
  // Inputs:
  //   CONSTRAINT_CALCULATOR is the object that will be used to evaluate
  //     the constraints.
  //   XM(Nw) and YM(Np) are the measured values of the well- and
  //     poorly-measured variables, respectively.
  //   X(Nw) and Y(Np) are the starting values for the fit.
  //   G_I(Nw,Nw) is the error matrix for the well-measured variables.
  //   Y(Np,Np) is the inverse error matrix for the poorly-measured variables.
  //
  // Outputs:
  //   X(Nw) and Y(Np) is the point at the minimum.
  //   PULLX(Nw) and PULLY(Np) are the pull quantities.
  //   Q(Nw,Nw), R(Np,Np), and S(Nw,Np) are the final error matrices
  //     between all the variables.
  //
  // The return value is the final chisq.  Returns a value < 0 if the
  // fit failed to converge.

  /**
     @brief Perform the \f$\chi^{2}\f$ constrained fit.


     @param constraint_calculator The object that will be used to evaluate
     the constraints.

     @param xm Measured values of the well-measured variables, has dimension
     <i>Nw</i>.

     @param x  Before the fit: starting value of the well-measured variables
     for the fit, has dimension <i>Nw</i>.
     After the fit: final values of the well-measured variables.

     @param ym Measured values of the poorly-measured variables, has
     dimension <i>Np</i>.

     @param y  Before the fit: starting value of the poorly-measured variables
     for the fit, has dimension <i>Np</i>.
     After the fit: final values of the poorly-measured variables.

     @param G_i Error matrix for the well-measured variables, has dimension
     <i>Nw,Nw</i>.

     @param Y Inverse error matrix for the poorly-measured variables, has
     dimension <i>Np,Np</i>.

     @param pullx Pull quantities for the well-measured variables, has
     dimension <i>Nw</i>.

     @param pully Pull quantities for the poorly-measured variables, has
     dimension <i>Np</i>.

     @param Q Error matrix for the well-measured variables, has dimension
     <i>Nw,Nw</i>.

     @param R Error matrix for the poorly-measured variables, has dimension
     <i>Np,Np</i>.

     @param S Error matrix for the correlation between well-measured variables
     and poorly-measured variables, has dimension <i>Nw,Np</i>.

     @par Input:
     <i>constraint_calculator, xm, x, ym, y, G_i, y</i>.

     @par Output:
     <i>x, y, pullx, pully, Q, R, S</i>.
     @par Return:
     \f$\chi^{2}\f$ of the fit.  Should returns a negative value if the fit
     does not converge.
   */
  virtual double fit (Constraint_Calculator& constraint_calculator,
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
                      Matrix& S) = 0;

  // Print out any internal state to S.
  /**
    @brief Print out internal state to output stream.

    @param s Output stream to which to write.

    @par Return:
    The stream <i>s</i>;

   */
  virtual std::ostream& print (std::ostream& s) const;

  // Print out internal state to S.
  friend std::ostream& operator<< (std::ostream& s, const Base_Constrainer& f);


private:
  // Parameter settings.

  /**
     Parameter settings for this instance of Base_Constrainer.
   */
  const Base_Constrainer_Args _args;


protected:
  // Helper function to evaluate the constraints.
  // This takes care of checking what the user function returns against
  // numerical derivatives, if that was requested.

  /**
     @brief Helper function to evaluate constraints.  This takes care of
     checking what the user function returns againts numerical
     derivatives, it that was requested.

     @par Purpose
     Call the constraint function for the point <i>(x,y)</i>.  Return
     <i>F</i>, <i>Bx</i>, <i>By</i>, and a boolean flag saying if the point
     is acceptable.  If <i>test_gradient</i> is on, we verify the gradients
     returned by also computing them numerically.

     @param constraint_calculator the User-supplied object to evaluate the
     constraints.

     @param x Vector of well-measured quantities where we evaluate the
     constraints.  Has dimension of <i>Nw</i>.

     @param y Vector of poorly-measured quantities where we evaluate the
     the constraints.  Has dimension of <i>Np</i>.

     @param F Results of the constraint functions.

     @param Bx Gradients of <i>F</i> with respect to <i>x</i>, has dimension
     <i>(Nw,Nc)</i>.

     @param By Gradients of <i>F</i> with respect to <i>y</i>, has dimension
     <i>(Np,Nc)</i>.

     @par Input:
     <i>constraint_calculator, x, y.</i>

     @par Output:
     <i>F, Bx, By.</i>

     @par Return:
     <b>true</b> if the point <i>(x,y)</i> is accepted.<br>
     <b>false</b> if the point <i>(x,y)</i> is rejected

   */
  bool call_constraint_fcn (Constraint_Calculator& constraint_calculator,
                            const Column_Vector& x,
                            const Column_Vector& y,
                            Row_Vector& F,
                            Matrix& Bx,
                            Matrix& By) const;
};


} // namespace hitfit


#endif // not HITFIT_BASE_CONSTRAINER_H

