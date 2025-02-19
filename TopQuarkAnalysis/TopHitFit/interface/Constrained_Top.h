//
// $Id: Constrained_Top.h,v 1.1 2011/05/26 09:46:53 mseidel Exp $
//
// File: hitfit/Constrained_Top.h
// Purpose: Do kinematic fitting for a ttbar -> ljets event.
// Created: Jul, 2000, sss, based on run 1 mass analysis code.
//
// CMSSW File      : interface/Constrained_Top.h
// Original Author : Scott Stuart Snyder <snyder@bnl.gov> for D0
// Imported to CMSSW by Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>


/**
    @file Constrained_Top.h

    @brief Do a constrained kinematic fit of a \f$t\bar{t}\to\ell +
    \rm{jets}\f$ event.

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


#ifndef HITFIT_CONSTRAINED_TOP_H
#define HITFIT_CONSTRAINED_TOP_H


#include "TopQuarkAnalysis/TopHitFit/interface/Fourvec_Constrainer.h"
#include "TopQuarkAnalysis/TopHitFit/interface/matutil.h"
#include <iosfwd>


namespace hitfit {


class Defaults;
class Lepjets_Event;

/**

    @class Constrained_Top_Args

    @brief Hold on to parameters for the Constrained_Top class.

 */
class Constrained_Top_Args
//
// Purpose: Hold on to parameters for the Constrained_Top class.
//
// Parameters controlling the operation of the fitter:
//   float bmass        - The mass to which b jets should be fixed.
//
{
public:
  // Constructor.  Initialize from a Defaults object.
  /**
     Constructor.

     @param defs An instance of defaults objects.  The instance must contain
     the variables necessary to set up a fourvec_constrainer_args, and the
     following variables with types and names.
     - double <i>bmass</i>.
     - bool <i>equal_side</i>.

   */
  Constrained_Top_Args (const Defaults& defs);

  // Retrieve parameter values.
  /**
     Return the <i>_bmass</i> parameter.
   */
  double bmass () const;

  // Arguments for subobjects.
  /**
     Return the <i>_fourvec_constrainer_args</i> parameter.
   */
  const Fourvec_Constrainer_Args& fourvec_constrainer_args () const;

  // Retrieve requirement for equal mass on both sides
  /**
     Return the <i>_equal_side</i> parameter.
   */
  bool equal_side() const;

private:
  // Hold on to parameter values.

  /**
     The mass to which \f$b\f$-quark jets should be fixed.
   */
  double _bmass;

  /**
     Arguments for the subobjects, constraints among the four vectors
     in the event.
   */
  Fourvec_Constrainer_Args _fourvec_constrainer_args;

  /**
     If true, requires that the leptonic side and hadronic side of
     \f$t\bar{t}\to\ell + \rm{jets}\f$ event to have equal mass.
   */
  bool _equal_side;

};


//*************************************************************************


/**
    @class Constrained_Top
    @brief Do a constrained kinematic fitting for a
    \f$t\bar{t}\to\ell + \rm{jets}\f$ event.
 */
class Constrained_Top
//
// Purpose: Do kinematic fitting for a ttbar -> ljets event.
//
{
public:
  // Constructor.
  // LEPW_MASS, HADW_MASS, and TOP_MASS are the masses to which
  // those objects should be constrained.  To remove a constraint,
  // set the mass to 0.
  /**
     @brief Constructor, create an instance of the Constrained_Top object
     from the arguments object and the mass constraints.

     @param args Argument for this instance of Constrained_Top object.

     @param lepw_mass The mass to which the leptonic W should be constrained.
     If this parameter is set to 0, the constraint is skied.

     @param hadw_mass The mass to which the hadronic W should be constrained.
     If this parameter is set to 0, the constraint is skied.

     @param top_mass The mass to which the top quarks should be constrained.
     If this parameter is set to 0, the constraints is skied.

   */
  Constrained_Top (const Constrained_Top_Args& args,
                   double lepw_mass,
                   double hadw_mass,
                   double top_mass);

  // Do a constrained fit.
  /**
     @brief Do a constrained fit of \f$t\bar{t}\to\ell + \rm{jets}\f$ events.
     Returns the
     top mass and its error in <i>mt</i> and <i>sigmt</i>, and the pull
     quantities in <i>pullx</i> and <i>pully</i>. Returns the \f$\chi^{2}\f$,
     this will be negative if the fit failed to converge.

     @param ev The event to be fitted (input), and the event after fitting
     (output).

     @param mt The fitted top mass.

     @param sigmt The uncertainty on the fitted top mass.

     @param pullx Pull quantities for the well-measured variables.

     @param pully Pull quantities for the poorly-measured variables.

     @par Input:
     - Lepjets_Event <i>ev</i>.

     @par Output:
     - Lepjets_Event <i>ev</i>.
     - double <i>mt</i>.
     - double <i>sigmt</i>.
     - Column_Vector <i>pullx</i>.
     - Column_Vector <i>pully</i>.

     @par Return:
     The \f$\chi^{2}\f$ of the fit.  Return a negative value if the fit
     didn't converge.

   */
  double constrain (Lepjets_Event& ev,
                    double& mt,
                    double& sigmt,
                    Column_Vector& pullx,
                    Column_Vector& pully);

  // Dump out our state.
  friend std::ostream& operator<< (std::ostream& s, const Constrained_Top& ct);


private:
  // Parameter settings.
  /**
     Parameter settings for the \f$\chi^{2}\f$ constrainer.
   */
  const Constrained_Top_Args _args;

  // The guy that actually does the work.
  /**
     The guy that actually does the work.
   */
  Fourvec_Constrainer _constrainer;
};


} // namespace hitfit


#endif // not HITFIT_CONSTRAINED_TOP_H
