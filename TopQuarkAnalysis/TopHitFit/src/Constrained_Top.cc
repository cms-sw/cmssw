//
// $Id: Constrained_Top.cc,v 1.1 2011/05/26 09:46:59 mseidel Exp $
//
// File: src/Constrained_Top.cc
// Purpose: Do kinematic fitting for a ttbar -> ljets event.
// Created: Jul, 2000, sss, based on run 1 mass analysis code.
//
// CMSSW File      : src/Constrained_Top.cc
// Original Author : Scott Stuart Snyder <snyder@bnl.gov> for D0
// Imported to CMSSW by Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>
//


/**

    @file Constrained_Top.cc

    @brief Do a constrained kinematic fit of a \f$t\bar{t}\to\ell +
    \rm{jets}\f$ event.  See the documentation for the header file
    Constrained_Top.h for details.

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

#include "TopQuarkAnalysis/TopHitFit/interface/Constrained_Top.h"
#include "TopQuarkAnalysis/TopHitFit/interface/Fourvec_Event.h"
#include "TopQuarkAnalysis/TopHitFit/interface/Lepjets_Event.h"
#include "TopQuarkAnalysis/TopHitFit/interface/Defaults.h"
#include <ostream>
#include <cassert>
#include <stdio.h>


using std::ostream;


namespace hitfit {


//*************************************************************************
// Argument handling.
//


Constrained_Top_Args::Constrained_Top_Args (const Defaults& defs)
//
// Purpose: Constructor.
//
// Inputs:
//   defs -        The Defaults instance from which to initialize.
//
  : _bmass (defs.get_float ("bmass")),
    _fourvec_constrainer_args (defs),
    _equal_side(defs.get_bool("equal_side"))

{
}


double Constrained_Top_Args::bmass () const
//
// Purpose: Return the bmass parameter.
//          See the header for documentation.
//
{
  return _bmass;
}


const Fourvec_Constrainer_Args&
Constrained_Top_Args::fourvec_constrainer_args () const
//
// Purpose: Return the contained subobject parameters.
//
{
  return _fourvec_constrainer_args;
}


bool Constrained_Top_Args::equal_side () const
//
// Purpose: Return the equal_side parameter
//          See the header for documentation.
//
{
  return _equal_side;
}


//*************************************************************************


Constrained_Top::Constrained_Top (const Constrained_Top_Args& args,
                                  double lepw_mass,
                                  double hadw_mass,
                                  double top_mass)
//
// Purpose: Constructor.
//
// Inputs:
//   args -        The parameter settings for this instance.
//   lepw_mass -   The mass to which the leptonic W should be constrained,
//                 or 0 to skip this constraint.
//   hadw_mass -   The mass to which the hadronic W should be constrained,
//                 or 0 to skip this constraint.
//   top_mass -    The mass to which the top quarks should be constrained,
//                 or 0 to skip this constraint.
//
  : _args (args),
    _constrainer (args.fourvec_constrainer_args())
{
  char buf[256];
  if (lepw_mass > 0) {
    sprintf (buf, "(%d %d) = %f", nu_label, lepton_label, lepw_mass);
    _constrainer.add_constraint (buf);
  }

  if (hadw_mass > 0) {
    sprintf (buf, "(%d %d) = %f", hadw1_label, hadw2_label, hadw_mass);
    _constrainer.add_constraint (buf);
  }

  if (args.equal_side()) {
      sprintf (buf, "(%d %d %d) = (%d %d %d)",
               nu_label, lepton_label, lepb_label,
               hadw1_label, hadw2_label, hadb_label);
      _constrainer.add_constraint (buf);
  }

  if (top_mass > 0) {
    sprintf (buf, "(%d %d %d) = %f",
             hadw1_label, hadw2_label, hadb_label, top_mass);
    _constrainer.add_constraint (buf);
  }
  else {
    sprintf (buf, "(%d %d %d) = 0", hadw1_label, hadw2_label, hadb_label);
    _constrainer.mass_constraint (buf);
  }
}

namespace {


/**

    @brief Helper function to create an object to put into Fourvec_Event.

    @param obj The input object.

    @param mass The mass to which it should be constrained.

    @param type The type to assign to it.

    @par Return:
    The constructed <i>FE_Obj</i>.

 */
FE_Obj make_fe_obj (const Lepjets_Event_Lep& obj, double mass, int type)
//
// Purpose: Helper to create an object to put into the Fourvec_Event.
//
// Inputs:
//   obj -         The input object.
//   mass -        The mass to which it should be constrained.
//   type -        The type to assign it.
//
// Returns:
//   The constructed FE_Obj.
//
{
  return FE_Obj (obj.p(), mass, type,
                 obj.p_sigma(), obj.eta_sigma(), obj.phi_sigma(),
                 obj.res().p_res().inverse());
}


/**

    @brief Convert from a Lepjets_Event to a Fourvec_Event.

    @param ev The input event.

    @param bmass The mass to which b-jets should be fixed.

    @param fe The output Fourvec_Event.

    @par Input:
    - Lepjets_Event <i>ev</i>.
    - double <i>bmass</i>.

    @par Output:
    - Fourvec_Event <i>fe</i>.

 */
void do_import (const Lepjets_Event& ev, double bmass, Fourvec_Event& fe)
//
// Purpose: Convert from a Lepjets_Event to a Fourvec_Event.
//
// Inputs:
//   ev -          The input event.
//   bmass -       The mass to which b-jets should be fixed.
//
// Outputs:
//   fe -          The initialized Fourvec_Event.
//
{
  assert (ev.nleps() == 1);
  fe.add (make_fe_obj (ev.lep(0), 0, lepton_label));

  bool saw_lepb = false;
  bool saw_hadb = false;
  for (std::vector<Lepjets_Event_Jet>::size_type j=0; j < ev.njets(); j++) {
    if (ev.jet(j).type() == isr_label || ev.jet(j).type() == higgs_label)
      continue;
    double mass = 0;
    if (ev.jet(j).type() == lepb_label && !saw_lepb) {
      mass = bmass;
      saw_lepb = true;
    }
    else if (ev.jet(j).type() == hadb_label && !saw_hadb) {
      mass = bmass;
      saw_hadb = true;
    }
    fe.add (make_fe_obj (ev.jet(j), mass, ev.jet(j).type()));
  }

  fe.set_nu_p (ev.met());
  Fourvec kt = ev.kt ();
  fe.set_kt_error (ev.kt_res().sigma (kt.x()),
                   ev.kt_res().sigma (kt.y()),
                   0);
}


/**

    @brief Convert from a Fourvec_Event to a Lepjets_Event.

    @param fe The input event.

    @param ev The returned Lepjets_Event.

    @par Input:
    - Fourvec_Event <i>fe</i>.
    - Lepjets_Event <i>ev</i>.

    @par Output:
    - Lepjets_Event <i>ev</i>

 */
void do_export (const Fourvec_Event& fe, Lepjets_Event& ev)
//
// Purpose: Convert from a Fourvec_Event to a Lepjets_Event.
//
// Inputs:
//   fe -          The input event.
//   ev -          The original Lepjets_Event.
//
// Outputs:
//   ev -          The updated Lepjets_Event.
//
{
  ev.lep(0).p() = fe.obj(0).p;
  for (std::vector<Lepjets_Event_Jet>::size_type j=0, k=1; j < ev.njets(); j++) {
    if (ev.jet(j).type() == isr_label || ev.jet(j).type() == higgs_label)
      continue;
    ev.jet(j).p() = fe.obj(k++).p;
  }

  ev.met() = fe.nu();
}


} // unnamed namespace


double Constrained_Top::constrain (Lepjets_Event& ev,
                                   double& mt,
                                   double& sigmt,
                                   Column_Vector& pullx,
                                   Column_Vector& pully)
//
// Purpose: Do a constrained fit for EV.  Returns the top mass and
//          its error in MT and SIGMT, and the pull quantities in PULLX and
//          PULLY.  Returns the chisq; this will be < 0 if the fit failed
//          to converge.
//
// Inputs:
//   ev -          The event we're fitting.
//
// Outputs:
//   ev -          The fitted event.
//   mt -          Requested invariant mass.
//   sigmt -       Uncertainty on mt.
//   pullx -       Pull quantities for well-measured variables.
//   pully -       Pull quantities for poorly-measured variables.
//
// Returns:
//   The fit chisq, or < 0 if the fit didn't converge.
//
{
  Fourvec_Event fe;
  do_import (ev, _args.bmass (), fe);
  double chisq = _constrainer.constrain (fe, mt, sigmt, pullx, pully);
  do_export (fe, ev);

  return chisq;
}


/**
    @brief Output stream operator, print the content of this Constrained_Top
    object to an output stream.

    @param s The output stream to which to write.

    @param ct The instance of Constrained_Top to be printed.

*/
std::ostream& operator<< (std::ostream& s, const Constrained_Top& ct)
//
// Purpose: Print the object to S.
//
// Inputs:
//   s -           The stream to which to write.
//   ct -          The object to write.
//
// Returns:
//   The stream S.
//
{
  return s << ct._constrainer;
}


} // namespace hitfit
