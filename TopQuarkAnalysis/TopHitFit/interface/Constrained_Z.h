//
// $Id: Constrained_Z.h,v 1.1 2011/05/26 09:46:53 mseidel Exp $
//
// File: Constrained_Z.h
// Purpose: Do kinematic fitting for a (Z->ll)+jets event.
// Created: Apr, 2004, sss
//
// CMSSW File      : interface/Constrained_Z.h
// Original Author : Scott Stuart Snyder <snyder@bnl.gov> for D0
// Imported to CMSSW by Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>


/**
    @file Constrained_Z.h

    @brief Do a constrained kinematic fit of a
    \f$Z\to\ell^{+}\ell^{-}+\rm{jets}\f$ event.

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


#ifndef HITFIT_CONSTRAINED_Z_H
#define HITFIT_CONSTRAINED_Z_H


#include "TopQuarkAnalysis/TopHitFit/interface/Fourvec_Constrainer.h"
#include "TopQuarkAnalysis/TopHitFit/interface/matutil.h"
#include <iosfwd>


namespace hitfit {


class Defaults;
class Lepjets_Event;


/**
    @class Constrained_Z_Args
    @brief Hold on to parameters for the Constrained_Z class.
 */
class Constrained_Z_Args
//
// Purpose: Hold on to parameters for the Constrained_Z class.
//
// Parameters controlling the operation of the fitter:
//   float zmass        - The mass to which the Z should be fixed.
//
{
public:
  // Constructor.  Initialize from a Defaults object.
  /**
     @brief Constructor, initialize from a Defaults object.

     @brief param defs An instance of the Defaults object.  The instance must
     the variables necessary to set up a fourvec_constrainer_args, and the
     following variables with types and names.
     - double <i>zmass</i>.
   */
  Constrained_Z_Args (const Defaults& defs);

  // Retrieve parameter values.
  /**
     Return the _zmass parameter.
   */
  double zmass () const;

  // Arguments for subobjects.
  /**
     Return the _fourvec_constrainer_args parameter.
   */
  const Fourvec_Constrainer_Args& fourvec_constrainer_args () const;

private:
  // Hold on to parameter values.
  /**
     The mass to which Z bosons should be fixed to.
   */
  double _zmass;

  /**
     Arguments for the subobjects, constraints among the four vectors
     in the event.
   */
  Fourvec_Constrainer_Args _fourvec_constrainer_args;
};


//*************************************************************************


/**
    @class Constrained_Z
    @brief Do a constrained kinematic fitting for a \f$Z\to\ell^{+}\ell^{-} +
    \rm{jets}\f$ event.
 */
class Constrained_Z
//
// Purpose: Do kinematic fitting for a (Z->ll)+jets event.
//
{
public:
  // Constructor.
  /**
     @brief Constructor, create an instance of the Constrained_Z object
     from the argument object.
     @param args Argument for this instance of Constrained_Z object.
   */
  Constrained_Z (const Constrained_Z_Args& args);

  // Do a constrained fit.
  /**
     @brief Do a constrained fit of \f$Z\to\ell^{+}\ell^{-} + \rm{jets}\f$
     event.  Returns the pull quantities in <i>pull</i>.  Returns
     the \f$\chi^{2}\f$, this will be negative if the fit failed to converge.
     @param ev The event to be fitted (input), and the event after fitting
     (output).
     @param pull Pull quantities for the well-measured variables.

     @par Input:
     - Lepjets_Event <i>ev</i>.

     @par Output:
     - Lepjets_Event <i>ev</i>.
     - Column_Vector <i>pull</i>.

     @par Return:
     The \f$\chi^{2}\f$ of the fit.  Return a negative value if the fit
     didn't converge.

  */
  double constrain (Lepjets_Event& ev, Column_Vector& pull);

  // Dump out our state.
  friend std::ostream& operator<< (std::ostream& s, const Constrained_Z& cz);


private:
  // Parameter settings.
  /**
     Parameter settings for the \f$\chi^{2}\f$ constrainer.
   */
  const Constrained_Z_Args& _args;

  // The guy that actually does the work.
  /**
     The guy that actually does the work.
   */
  Fourvec_Constrainer _constrainer;
};


} // namespace hitfit


#endif // not HITFIT_CONSTRAINED_Z_H
