//
// $Id: Fourvec_Constrainer.h,v 1.1 2011/05/26 09:46:53 mseidel Exp $
//
// File: hitfit/Fourvec_Constrainer.h
// Purpose: Do a kinematic fit for a set of 4-vectors, given a set
//          of mass constraints.
// Created: Jul, 2000, sss, based on run 1 mass analysis code.
//
// Do a constrained fit on a set of 4-vectors.
// The input is in a Fourvec_Event instance.  This consists
// of a collection of objects, each of which has a 4-momentum
// and an integer label.  (Also uncertainties, etc.)
//
// We also have a set of mass constraints, based on sums of objects
// with specified labels.  A constraint can either require that the
// invariant mass of a set of objects is constant, or that the masses
// of two sets be equal to each other.  For example, the constraint
//
//    (1 2) = 80
//
// says that the sum of all objects with labels 1 and 2 should have
// a mass of 80.  And the constraint
//
//    (1 2) = (3 4)
//
// says that the sum of all objects with labels 1 and 2 should
// have an invariant mass the same as the sum of all objects with
// labels 3 and 4.
//
// All the objects are fixed to constant masses for the fit.
// (These masses are attributes of the objects in the Fourvec_Event.)
// This is done by scaling either the 4-vector's 3-momentum or energy,
// depending on the setting of the `use_e' parameter.
//
// If there is no neutrino present in the event, two additional constraints
// are automatically added for total pt balance, unless the parameter
// ignore_met has been set.
//
// When the fit completes, this object can compute an invariant mass
// out of some combination of the objects, including an uncertainty.
// The particular combination is specified through the method mass_contraint();
// it gets a constraint string like normal; the lhs of the constraint
// is the mass that will be calculated.  (The rhs should be zero.)
//
// CMSSW File      : interface/Fourvec_Constrainer.h
// Original Author : Scott Stuart Snyder <snyder@bnl.gov> for D0
// Imported to CMSSW by Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>
//


/**

    @file Fourvec_Constrainer.h

    @brief Do a kinematic fit for a set of four-vectors, given a set
    of mass constraints.

    @author Scott Stuart Snyder <snyder@bnl.gov>

    @par Creation date:
    Jul 2000.

    @par Modification History:
    Apr 2009: Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>:
    Imported to CMSSW.<br>
    Nov 2009: Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>:
    Added doxygen tags for automatic generation of documentation.

    @par Terms of Usage:
    With consent for the original author (Scott Snyder).

*/

#ifndef HITFIT_FOURVEC_CONSTRAINER_H
#define HITFIT_FOURVEC_CONSTRAINER_H


#include <string>
#include <vector>
#include <iosfwd>
#include "TopQuarkAnalysis/TopHitFit/interface/Constraint.h"
#include "TopQuarkAnalysis/TopHitFit/interface/Chisq_Constrainer.h"
#include "TopQuarkAnalysis/TopHitFit/interface/matutil.h"


namespace hitfit {


class Defaults;
class Fourvec_Event;
class Pair_Table;


/**
    @class Fourvec_Constrainer_Args

    @brief Hold on to parameters for the Fourvec_Constrainer class.
 */
class Fourvec_Constrainer_Args
//
// Purpose: Hold on to parameters for the Fourvec_Constrainer class.
//
// Parameters controlling the operation of the fitter:
//   bool use_e         - If true, then when rescaling the 4-vectors
//                        for a given mass, keep the measured energy
//                        and scale the 3-momentum.  Otherwise, keep
//                        the 3-momentum and scale the energy.
//   float e_com        - The center-of-mass energy.
//                        (Used only to keep the fit from venturing
//                        into completely unphysical regions.)
//   bool ignore_met    - If this is true and the event does not
//                        have a neutrino, then the fit will be done
//                        without the overall transverse momentum
//                        constraint (and thus the missing Et information
//                        will be ignored).  If the event does have
//                        a neutrino, this parameter is ignored.
//
{
public:
  // Constructor.  Initialize from a Defaults object.
  /**
     @brief Constructor, initialize from a Defaults object.

     @param defs The set of parameters for the initialization.  The instance
     must contain the following variables of types and names:
     - bool <i>use_e</i>.
     - double <i>e_com</i>.
     - bool <i>ignore_met</i>.
   */
  Fourvec_Constrainer_Args (const Defaults& defs);

  // Retrieve parameter values.
  /**
     Return the <i>_use_e</i> parameter.
   */
  bool use_e () const;

  /**
     Return the <i>_e_com</i> parameter.
   */
  double e_com () const;

  /**
     Return the <i>_ignore_met</i> parameter.
   */
  bool ignore_met () const;


  // Arguments for subobjects.
  const Chisq_Constrainer_Args& chisq_constrainer_args () const;


private:
  // Hold on to parameter values.
  /**
     If TRUE, then when rescaling a four-momentum for a given mass,
     keep the measured energy and scale the three-momentum.
     If FALSE, then keep the three-momentum and scale the energy.
   */
  bool _use_e;

  /**
     The center-of-mass energy. Used only to keep the fit from
     going into a completely unphysical region.
   */
  double _e_com;

  /**
     If TRUE and the event does not have a neutrino, then the fit
     will be done without the overall transverse momentum constraint.
     Thus the missing transverse energy information will be ignored.
     If the event does have a neutrino, this parameter is ignored.
     If FALSE, take into account the overall transverse momentum constraint.
   */
  bool _ignore_met;

  /**
     The internal arguments for the subobjects.
   */
  Chisq_Constrainer_Args _chisq_constrainer_args;
};


//*************************************************************************


/**
    @class Fourvec_Constrainer

    @brief Do a kinematic fit for a set of four-momenta, given a set
    of mass constraints.

    The input is in a Fourvec_Event instance.  This consists of a collection
    of objects, each of which has a four-momentum, an uncertainty on the
    four-momentum, and an integer label.

    There are also a set of mass constraints, based on sums of objects
    with specified labels.  A constraint can either require that the invariant
    mass of a set of objects is constant, or that the mass of two sets of
    objects be equal to each other.  For example, the constraint

    \f[ (1~~2) = 80 \f]

    says that the sum of all objects with labels 1 and 2 should have a mass
    of 80, where the unit of mass is always in GeV.  And the constraint

    \f[ (1~~2) = (3~~4) \f]

    says that the sum of all objects with labels 1 and 2 should have an
    invariant mass equal as the invariant mass of the sum of all objects with
    label 3 and 4.

    All the objects are fixed to constant masses for the fit.  These masses
    are attributes of the objects in the Fourvec_Event.  This is done by
    scaling either the four-momentum's three-momentum or energy,
    depending on the setting of the <i>use_e</i> parameter.

    If there is no neutrino present in the event, two additional
    constraints are automatically added for total \f$p_{T}\f$ balance, unless
    the parameter <i>ignore_met</i> has been set to TRUE.

    When the fit completes, this object can  compute an invariant mass out
    of some combination of the objects, including and uncertainty.
    The particular combination is specified through the method
    mass_constraint((); it gets a constraint string like normal with
    the LHS of the constraint i s the mass which will be calculater, and
    the RHS of the oncstrant should be zero.

 */
class Fourvec_Constrainer
//
// Purpose: Do a kinematic fit for a set of 4-vectors, given a set
//          of mass constraints.
//
{
public:
  // Constructor.
  // ARGS holds the parameter settings for this instance.
  /**
     @brief Constructor.

     @param args Parameter settings for this instance.
   */
  Fourvec_Constrainer (const Fourvec_Constrainer_Args& args);

  // Specify an additional constraint S for the problem.
  // The format for S is described above.
  /**
     @brief Specify an additional constraint <i>s</i> for the problem.
     The format for <i>s</i> is described in the class description.

     @param s The constraint to add.
   */
  void add_constraint (std::string s);

  // Specify the combination of objects that will be returned by
  // constrain() as the mass.  The format of S is the same as for
  // normal constraints.  The LHS specifies the mass to calculate;
  // the RHS should be zero.
  // This should only be called once.
  /**
     @brief Specify a combination of objects that will be returned
     by the constrain() method as mass.  The format of <i>s</i>
     is the same as for normal constraints.  The left-hand side specifies
     the mass to calculate, the right-hand side should be zero.
     This combination of objects will be called only once.

     @param s The constraint defining the mass.
   */
  void mass_constraint (std::string s);

  // Do a constrained fit for EV.  Returns the requested mass and
  // its error in M and SIGM, and the pull quantities in PULLX and
  // PULLY.  Returns the chisq; this will be < 0 if the fit failed
  // to converge.
  /**
     @brief Do a constrained fit for event <i>ev</i>.  Returns the requested
     mass and its uncertainty in <i>m</i> and <i>sigm</i>, and the pull
     quantities in <i>pullx</i> and <i>pully</i>.  Returns the \f$\chi^{2}\f$,
     the value will be negative if the fit failed to converge.

     @param ev The event to be fitted (input), and later after fitting
     (output).

     @param m The requested invariant mass to calculate.

     @param sigm The uncertainty in the requested invariant mass.

     @param pullx Pull quantities for well-measured variables.

     @param pully Pull quantities for poorly-measured variables.

     @par Input:
     - <i>ev</i> (before fit).

     @par Output:
     - <i>ev</i> (after fit).
     - <i>m</i>.
     - <i>sigm</i>.
     - <i>pullx</i>
     - <i>pully</i>.
     @par Returns:
     The fit \f$\chi^{2}\f$, this value will be negative if the fit failed
     to converge.
   */
  double constrain (Fourvec_Event& ev,
                    double& m,
                    double& sigm,
                    Column_Vector& pullx,
                    Column_Vector& pully);

  // Dump the internal state.
  friend std::ostream& operator<< (std::ostream& s,
                                   const Fourvec_Constrainer& c);


private:
  // Parameter settings.
  /**
     Parameter settings for this instance.
   */
  const Fourvec_Constrainer_Args _args;

  // The constraints for this problem.
  /**
     The constraints for this problem.
   */
  std::vector<Constraint> _constraints;

  // The constraint giving the mass to be calculated.  This
  // should have no more than one entry.
  /**
      The constraint giving the mass to be calculated.  This should
      have no more than one entry.
   */
  std::vector<Constraint> _mass_constraint;
};


} // namespace hitfit


#endif // not HITFIT_FOURVEC_CONSTRAINER_H
