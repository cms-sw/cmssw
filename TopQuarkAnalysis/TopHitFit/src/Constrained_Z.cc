//
// $Id: Constrained_Z.cc,v 1.1 2011/05/26 09:46:59 mseidel Exp $
//
// File: Constrained_Z.cc
// Purpose: Do kinematic fitting for a (Z->ll)+jets event.
// Created: Apr, 2004, sss
//
// CMSSW File      : src/Constrained_Z.cc
// Original Author : Scott Stuart Snyder <snyder@bnl.gov> for D0
// Imported to CMSSW by Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>
//

/**
    @file Constrained_Z.cc

    @brief Do a constrained kinematic fit of a
    \f$Z\to\ell^{+}\ell^{-}+\rm{jets}\f$ event.
    See the documentation for the header file Constrained_Z.h for details.

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


#include "TopQuarkAnalysis/TopHitFit/interface/Constrained_Z.h"
#include "TopQuarkAnalysis/TopHitFit/interface/Fourvec_Event.h"
#include "TopQuarkAnalysis/TopHitFit/interface/Lepjets_Event.h"
#include "TopQuarkAnalysis/TopHitFit/interface/Defaults.h"
#include <ostream>
#include <cassert>
#include <stdio.h>


namespace hitfit {


//*************************************************************************
// Argument handling.
//


Constrained_Z_Args::Constrained_Z_Args (const Defaults& defs)
//
// Purpose: Constructor.
//
// Inputs:
//   defs -        The Defaults instance from which to initialize.
//
  : _zmass (defs.get_float ("zmass")),
    _fourvec_constrainer_args (defs)
{
}


double Constrained_Z_Args::zmass () const
//
// Purpose: Return the zmass parameter.
//          See the header for documentation.
//
{
  return _zmass;
}


const Fourvec_Constrainer_Args&
Constrained_Z_Args::fourvec_constrainer_args () const
//
// Purpose: Return the contained subobject parameters.
//
{
  return _fourvec_constrainer_args;
}


//*************************************************************************


Constrained_Z::Constrained_Z (const Constrained_Z_Args& args)
//
// Purpose: Constructor.
//
// Inputs:
//   args -        The parameter settings for this instance.
//   
  : _args (args),
    _constrainer (args.fourvec_constrainer_args())
{
  char buf[256];
  sprintf (buf, "(%d) = %f", lepton_label, _args.zmass());
  _constrainer.add_constraint (buf);
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

    @param fe The output Fourvec_Event.

    @par Input:
    - Lepjets_Event <i>ev</i>.

    @par Output:
    - Fourvec_Event <i>fe</i>.

 */
void do_import (const Lepjets_Event& ev, Fourvec_Event& fe)
//
// Purpose: Convert from a Lepjets_Event to a Fourvec_Event.
//
// Inputs:
//   ev -          The input event.
//
// Outputs:
//   fe -          The initialized Fourvec_Event.
//
{
  assert (ev.nleps() >= 2);
  fe.add (make_fe_obj (ev.lep(0), 0, lepton_label));
  fe.add (make_fe_obj (ev.lep(1), 0, lepton_label));

  for (std::vector<Lepjets_Event_Jet>::size_type j=0; j < ev.njets(); j++)
    fe.add (make_fe_obj (ev.jet(j), 0, isr_label));

  Fourvec kt = ev.kt ();
  fe.set_kt_error (ev.kt_res().sigma (kt.x()),
                   ev.kt_res().sigma (kt.y()),
                   0);
  fe.set_x_p (ev.met());
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
  ev.lep(1).p() = fe.obj(1).p;

  for (std::vector<Lepjets_Event_Jet>::size_type j=0, k=1; j < ev.njets(); j++)
    ev.jet(j).p() = fe.obj(k++).p;

  Fourvec nu;
  ev.met() = nu;
}


} // unnamed namespace


double Constrained_Z::constrain (Lepjets_Event& ev, Column_Vector& pull)
//
// Purpose: Do a constrained fit for EV.
//          Returns the pull quantities in PULL.
//          Returns the chisq; this will be < 0 if the fit failed
//          to converge.
//
// Inputs:
//   ev -          The event we're fitting.
//
// Outputs:
//   ev -          The fitted event.
//   pull -        Pull quantities for well-measured variables.
//
// Returns:
//   The fit chisq, or < 0 if the fit didn't converge.
//
{
  Fourvec_Event fe;
  do_import (ev, fe);
  Column_Vector pully;
  double m, sigm;
  double chisq = _constrainer.constrain (fe, m, sigm, pull, pully);
  do_export (fe, ev);

  return chisq;
}


  /**
     @brief Output stream operator, print the content of this Constrained_Z
     to an output stream.

     @param s The output stream to which to wrire.
     @param cz The instance of Constrained_Z to be printed.
   */
std::ostream& operator<< (std::ostream& s, const Constrained_Z& cz)
//
// Purpose: Print the object to S.
//
// Inputs:
//   s -           The stream to which to write.
//   cz -          The object to write.
//
// Returns:
//   The stream S.
//
{
  return s << cz._constrainer;
}


} // namespace hitfit
