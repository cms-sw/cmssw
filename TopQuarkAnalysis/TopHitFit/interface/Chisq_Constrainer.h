//
// $Id: Chisq_Constrainer.h,v 1.2 2011/10/13 09:49:48 snaumann Exp $
//
// File: hitfit/Chisq_Constrainer.h
// Purpose: Minimize a chisq subject to a set of constraints.
//          Based on the SQUAW algorithm.
// Created: Jul, 2000, sss, based on run 1 mass analysis code.
//
// For full details on the algorithm, see
//
//    @phdthesis{sssthesis,
//      author =       "Scott Snyder",
//      school =       "State University of New York at Stony Brook",
//      month =        may,
//      year =         "1995 (unpublished)"}
//    @comment{  note =         "available from {\tt http://www-d0.fnal.gov/publications\_talks/thesis/ snyder/thesis-ps.html}"
//    }
//
// CMSSW File      : interface/Chisq_Constrainer.h
// Original Author : Scott Stuart Snyder <snyder@bnl.gov> for D0
// Imported to CMSSW by Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>

/**

    @file Chisq_Constrainer.h

    @brief Minimize a \f$\chi^{2}\f$ subject to a set of constraints,
    based on the SQUAW algorithm.

    For the full details of the algorithm see:
    - Scott Snyder, <i>Measurement of the top quark mass at D0</i>, PhD thesis,
    State University of New York at Stony Brook, (1995).  Available from
    <a href=
    http://lss.fnal.gov/archive/thesis/1900/fermilab-thesis-1995-27.shtml>
    http://lss.fnal.gov/archive/thesis/1900/fermilab-thesis-1995-27.shtml</a>.

    - B. Abbott <i>et al.</i> (D0 Collaboration), <i>Direct Measurement of
    the Top Quark Mass at D0</i>,
    <a href="http://prola.aps.org/abstract/PRD/v58/i5/e052001">
    Phys. Rev. <b>D58</b>, 052001  (1998)</a>,
    <a href="http://arxiv.org/abs/hep-ex/9801025">arXiv:hep-ex/9801025</a>.

    - O. I. Dahl, T. B. Day, F. T. Solnitz, and M. L. Gould, <i>SQUAW Kinematic
    Fitting Program</i>, Group A Programming  Note P-126, Lawrence Radiation
    Laboratory (1968).  Available from
    <a href="http://alvarezphysicsmemos.lbl.gov">
    Luis Alvarez Physics Memos</a>.

    @par Creation date.
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


#ifndef HITFIT_CHISQ_CONSTRAINER_H
#define HITFIT_CHISQ_CONSTRAINER_H

#include "TopQuarkAnalysis/TopHitFit/interface/Base_Constrainer.h"
#include "TopQuarkAnalysis/TopHitFit/interface/matutil.h"
#include <iosfwd>


namespace hitfit {


class Defaults;


/**
     @class Chisq_Constrainer_Args
     @brief Hold on to parameters for the Chisq_Constrainer class.
*/
class Chisq_Constrainer_Args
//
// Purpose: Hold on to parameters for the Chisq_Constrainer class.
//
// Parameters controlling the operation of the fitter:
//   bool printfit      - If true, print a trace of the fit to cout.
//   bool use_G         - If true, check the chisq formula by computing
//                        chisq directly from G.  This requires that G_i
//                        be invertable.
//
// Parameters affecting the fit:
//   float constraint_sum_eps - Convergence threshold for sum of constraints.
//   float chisq_diff_eps - onvergence threshold for change in chisq.
//   int maxit          - Maximum number of iterations permitted.
//   int max_cut        - Maximum number of cut steps permitted.
//   float cutsize      - Fraction by which to cut steps.
//   float min_tot_cutsize - Smallest fractional cut step permitted.
//
// Parameters affecting testing modes:
//   float chisq_test_eps - When use_G is true, the maximum relative
//                          difference permitted between the two chisq
//                          calculations.
//
{
public:
  // Constructor.  Initialize from a Defaults object.

  /**
       @brief Constructor, creates an instance of Chisq_Constrainer_Args
       from a Defaults object.

       @param defs The Defaults instance from which to instantiate.
       The instance
       must contain variables of type and name:
       - bool <i>use_G</i>.
       - bool <i>printfit</i>.
       - double <i>constraint_sum_eps</i>.
       - double <i>chisq_diff_eps</i>.
       - int <i>maxit</i>.
       - int <i>max_cut</i>.
       - double <i>cutsize</i>.
       - double <i>min_tot_cutsize</i>.
       - double <i>chisq_test_eps</i>.
   */
  Chisq_Constrainer_Args (const Defaults& defs);

  // Retrieve parameter values.

  /**
     Return the <i>printfit</i> parameter.
   */
  bool printfit () const;

  /**
     Return the <i>use_G</i> parameter.
   */
  bool use_G () const;

  /**
     Return the <i>constraint_sum_eps</i> parameter.
   */
  double constraint_sum_eps () const;

  /**
     Return the <i>chisq_diff_eps</i> parameter.
   */
  double chisq_diff_eps () const;

  /**
     Return the <i>maxit</i> parameter.
   */
  unsigned  maxit () const;

  /**
     Return the <i>max_cut</i> parameter.
   */
  unsigned  max_cut () const;

  /**
     Return the <i>cutsize</i> parameter.
   */
  double cutsize () const;

  /**
     Return the <i>min_tot_cutsize</i> parameter.
   */
  double min_tot_cutsize () const;

  /**
     Return the <i>chisq_test_eps</i> parameter.
   */
  double chisq_test_eps () const;

  // Arguments for subobjects.

  /**
     Return the argument for the Base_Constrainer class.
   */
  const Base_Constrainer_Args& base_constrainer_args () const;


private:
  // Hold on to parameter values.

  /**
     If true, print a trace of the fit to std::cout.
   */
  bool _printfit;

  /**
     If true, check the \f$\chi^{2}\f$ formula by computing the \f$\chi^{2}\f$
     directly from \f${\bf G}\f$. This requires that \f${\bf G}_{i}\f$ be
     invertible.
   */
  bool _use_G;

  /**
     Convergence threshold for sum of constraints.
   */
  double _constraint_sum_eps;

  /**
     Convergence threshold for change in \f$\chi^{2}\f$.
   */
  double _chisq_diff_eps;

  /**
     Maxium number of iterations permitted.
   */
  int  _maxit;

  /**
     Maximum number of cut steps permitted.
   */
  int  _max_cut;

  /**
     Fraction by which to cut steps.
   */
  double _cutsize;

  /**
     Smallest fractional cut step permitted.
   */
  double _min_tot_cutsize;

  /**
     When <i>use_G</i> is true, the maximum relative difference between
     the \f$\chi^{2}\f$ calculations.
   */
  double _chisq_test_eps;


  /**
     Parameters for the underlying base class Base_Constrainer.
   */
  const Base_Constrainer_Args _base_constrainer_args;
};


//*************************************************************************


/**
    @class Chisq_Constrainer
    @brief Minimize a \f$\chi^{2}\f$ subject to a set of constraints.  Based
    on the SQUAW algorithm.
 */
class Chisq_Constrainer
//
// Purpose: Minimize a chisq subject to a set of constraints.
//          Based on the SQUAW algorithm.
//
  : public Base_Constrainer
{
public:
  // Constructor, destructor.
  // ARGS holds the parameter settings for this instance.

  /**
     Constructor.  Create an instance of Chisq_Constrainer from a
     Chisq_Constrainer_Args object.
     @param args The parameter settings for this instance.

   */
  Chisq_Constrainer (const Chisq_Constrainer_Args& args);

  /**
     Destructor.
   */
  virtual ~Chisq_Constrainer () {}

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
     @ brief Do a constrained fit.  Call the number of well-measured variables
     <i>Nw</i>, the number of poorly-measured variables <i>Np</i>,
     and the number of constraints <i>Nc</i>.

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
     <i>constraint_calculator, xm, x, ym, y, G_i, y.</i>.

     @par Output:
     <i>x, y, pullx, pully, Q, R, S.</i>
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
                      Matrix& S);

  // Print out any internal state to S.
/**
     @brief Print the state of this instance of Chisq_Constrainer.

     @param s The output stream to which the output is sent.
 */
  virtual std::ostream& print (std::ostream& s) const;


private:
  // Parameter settings.
  /**
         Parameter settings for this instance of Chisq_Constrainer.
   */
  const Chisq_Constrainer_Args _args;
};

} // namespace hitfit


#endif // not HITFIT_CHISQ_CONSTRAINER_H


