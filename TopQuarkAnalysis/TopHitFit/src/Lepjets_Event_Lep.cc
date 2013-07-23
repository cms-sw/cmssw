//
// $Id: Lepjets_Event_Lep.cc,v 1.1 2011/05/26 09:47:00 mseidel Exp $
//
// File: src/Lepjets_Event_Lep.cc
// Purpose: Represent a `lepton' in a Lepjets_Event.
// Created: Jul, 2000, sss, based on run 1 mass analysis code.
//
// CMSSW File      : src/Lepjets_Event_Lep.cc
// Original Author : Scott Stuart Snyder <snyder@bnl.gov> for D0
// Imported to CMSSW by Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>
//


/**
    @file Lepjets_Event_Lep.cc

	@brief Represent a lepton in an instance of Lepjets_Event class.
    See the header file Lepjets_Event_Lep for details.

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

#include "TopQuarkAnalysis/TopHitFit/interface/Lepjets_Event_Lep.h"


namespace hitfit {


Lepjets_Event_Lep::Lepjets_Event_Lep (const Fourvec& p,
                                      int type,
                                      const Vector_Resolution& res)
//
// Purpose: Constructor.
//
// Inputs:
//   p -           The 4-momentum.
//   type -        The type code.
//   res -         The resolution.
//
  : _p (p),
    _type (type),
    _res (res)
{
}


Fourvec& Lepjets_Event_Lep::p ()
//
// Purpose: Access the 4-momentum.
//
// Returns:
//   The 4-momentum.
//
{
  return _p;
}


const Fourvec& Lepjets_Event_Lep::p () const
//
// Purpose: Access the 4-momentum.
//
// Returns:
//   The 4-momentum.
//
{
  return _p;
}


int& Lepjets_Event_Lep::type ()
//
// Purpose: Access the type code.
//
// Returns:
//   The type code.
//
{
  return _type;
}


int Lepjets_Event_Lep::type () const
//
// Purpose: Access the type code.
//
// Returns:
//   The type code.
//
{
  return _type;
}


const Vector_Resolution& Lepjets_Event_Lep::res () const
//
// Purpose: Access the resolutions.
//
// Returns:
//   The resolutions.
//
{
  return _res;
}


Vector_Resolution& Lepjets_Event_Lep::res ()
//
// Purpose: Access the resolutions.
//
// Returns:
//   The resolutions.
//
{
  return _res;
}


double Lepjets_Event_Lep::p_sigma () const
//
// Purpose: Return the momentum (or 1/p) resolution for this object.
//
// Returns:
//   The momentum (or 1/p) resolution for this object.
//
{
  return _res.p_sigma (_p);
}


double Lepjets_Event_Lep::eta_sigma () const
//
// Purpose: Return the eta resolution for this object.
//
// Returns:
//   The eta resolution for this object.
//
{
  return _res.eta_sigma (_p);
}


double Lepjets_Event_Lep::phi_sigma () const
//
// Purpose: Return the phi resolution for this object.
//
// Returns:
//   The phi resolution for this object.
//
{
  return _res.phi_sigma (_p);
}


void Lepjets_Event_Lep::smear (CLHEP::HepRandomEngine& engine,
                               bool smear_dir /*= false*/)
//
// Purpose: Smear this object according to its resolutions.
//
// Inputs:
//   engine -      The underlying RNG.
//   smear_dir -   If false, smear the momentum only.
//
{
  _res.smear (_p, engine, smear_dir);
}


std::ostream& Lepjets_Event_Lep::dump (std::ostream& s,
                                       bool full /*= false*/) const
//
// Purpose: Dump out this object.
//
// Inputs:
//   s -           The stream to which to write.
//   full -        If true, dump the resolutions too.
//
// Returns:
//   The stream S.
//
{
    s << "[" << _type << "] " << _p << "; pt: " << _p.perp() << ", eta: " << _p.eta() << ", phi: " << _p.phi() ;
  if (full) {
    s << "\n    " << _res;
  }
  return s;
}


/**
    @brief Output stream operator, print the content of this Lepjets_Event_Lep
    to an output stream.

    @param s The stream to which to write.

    @param l The instance of Lepjets_Event_Lep to be printed.
 */
std::ostream& operator<< (std::ostream& s, const Lepjets_Event_Lep& l)
//
// Purpose: Dump out this object.
//
// Inputs:
//   s -           The stream to which to write.
//   l -           The object to dump.
//
// Returns:
//   The stream S.
//
{
  return l.dump (s);
}


bool Lepjets_Event_Lep::operator< (const Lepjets_Event_Lep& x) const
//
// Purpose: Sort objects by pt.
//
// Retruns:
//   True if this object's pt is less than that of x.
{
  return p().perp2() < x.p().perp2();
}


} // namespace hitfit
