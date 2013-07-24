//
// $Id: Lepjets_Event_Jet.cc,v 1.1 2011/05/26 09:47:00 mseidel Exp $
//
// File: src/Lepjets_Event_Jet.cc
// Purpose: Represent a `jet' in a Lepjets_Event.
// Created: Jul, 2000, sss, based on run 1 mass analysis code.
//
// CMSSW File      : src/Lepjets_Event_Jet.cc
// Original Author : Scott Stuart Snyder <snyder@bnl.gov> for D0
// Imported to CMSSW by Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>
//


/**
    @file Lepjets_Event_Jet.cc

    @brief Represent a jet in an instance of Lepjets_Event class.
    See the documentation for the header file Lepjets_Event_Jet.h for details.

    @author Scott Stuart Snyder <snyder@bnl.gov>

    @par Creation date:
    July 2000.

    @par Modification History:
    Apr 2009: Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>:
    Imported to CMSSW.<br>
    Nov 2009: Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>:
    Added doxygen tags for automatic generation of documentation.

    @par Terms of Usage:
    With consent for the original author (Scott Snyder).

 */

#include "TopQuarkAnalysis/TopHitFit/interface/Lepjets_Event_Jet.h"


namespace hitfit {


Lepjets_Event_Jet::Lepjets_Event_Jet (const Fourvec& p,
                                      int type,
                                      const Vector_Resolution& res,
                                      bool svx_tag /*= false*/,
                                      bool slt_tag /*= false*/,
                                      const Fourvec& tag_lep /*= Fourvec()*/,
                                      double slt_edep /*= 0*/)
//
// Purpose: Constructor.
//
// Inputs:
//   p -           The 4-momentum.
//   type -        The type code.
//   res -         The resolution.
//   svx_tag -     SVX tag flag.
//   slt_tag -     SLT tag flag.
//   tag_lep -     SLT lepton 4-momentum.
//   slt_edep -    SLT lepton energy deposition.
//
  : Lepjets_Event_Lep (p, type, res),
    _svx_tag (svx_tag),
    _slt_tag (slt_tag),
    _tag_lep (tag_lep),
    _slt_edep (slt_edep),
    _e0 (p.e())
{
}


bool Lepjets_Event_Jet::svx_tag () const
//
// Purpose: Access the SVX tag flag.
//
// Returns:
//   The SVX tag flag.
//
{
  return _svx_tag;
}


bool& Lepjets_Event_Jet::svx_tag ()
//
// Purpose: Access the SVX tag flag.
//
// Returns:
//   The SVX tag flag.
//
{
  return _svx_tag;
}


bool Lepjets_Event_Jet::slt_tag () const
//
// Purpose: Access the SLT tag flag.
//
// Returns:
//   The SLT tag flag.
//
{
  return _slt_tag;
}


bool& Lepjets_Event_Jet::slt_tag ()
//
// Purpose: Access the SLT tag flag.
//
// Returns:
//   The SLT tag flag.
//
{
  return _slt_tag;
}


const Fourvec& Lepjets_Event_Jet::tag_lep () const
//
// Purpose: Access the tag lepton 4-momentum.
//
// Returns:
//   The tag lepton 4-momentum.
//
{
  return _tag_lep;
}


Fourvec& Lepjets_Event_Jet::tag_lep ()
//
// Purpose: Access the tag lepton 4-momentum.
//
// Returns:
//   The tag lepton 4-momentum.
//
{
  return _tag_lep;
}


double Lepjets_Event_Jet::slt_edep () const
//
// Purpose: Access the tag lepton energy deposition.
//
// Returns:
//   The tag lepton energy deposition.
//
{
  return _slt_edep;
}


double& Lepjets_Event_Jet::slt_edep ()
//
// Purpose: Access the tag lepton energy deposition.
//
// Returns:
//   The tag lepton energy deposition.
//
{
  return _slt_edep;
}


double Lepjets_Event_Jet::e0 () const
//
// Purpose: Access the uncorrected jet energy.
//
// Returns:
//   The uncorrected jet energy.
//
{
  return _e0;
}


double& Lepjets_Event_Jet::e0 ()
//
// Purpose: Access the uncorrected jet energy.
//
// Returns:
//   The uncorrected jet energy.
//
{
  return _e0;
}


std::ostream& Lepjets_Event_Jet::dump (std::ostream& s,
                                       bool full /*= false*/) const
//
// Purpose: Dump out this object.
//
// Inputs:
//   s -           The stream to which to write.
//   full -        If true, dump all information for this object.
//
// Returns:
//   The stream S.
//
{
  Lepjets_Event_Lep::dump (s, full);
  if (_svx_tag)
    s << " (svx)";
  if (_slt_tag)
    s << " (slt)";
  if (full) {
    if (_slt_tag) {
      s << "    tag lep: " << _tag_lep;
      s << " edep: " << _slt_edep;
    }
    s << "\n";
  }
  return s;
}


/**
    @brief Output stream operator, print the content of this Lepjets_Event_Jet
    to an output stream.

    @param s The stream to which to write.

    @param l The instance of Lepjets_Event_Jet to be printed.
 */
std::ostream& operator<< (std::ostream& s, const Lepjets_Event_Jet& l)
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


char
jetTypeChar(int j)
//
// Purpose: Translate numeric jet type into char
//
// Inputs:
//   j -          jet type in integer
//
// Returns:
//   the jet type in char
//
{


    switch (j) {

    case hitfit::isr_label:
        return 'g';
    case hitfit::lepb_label:
        return 'b';
    case hitfit::hadb_label:
        return 'B';
    case hitfit::hadw1_label:
        return 'W';
    case hitfit::hadw2_label:
        return 'W';
    case hitfit::higgs_label:
        return 'h';
    case hitfit::unknown_label:
        return '?';
    default:
        return '?';
    }

    return '?';

}

std::string
jetTypeString(int j)
//
// Purpose: Translate numeric jet type into string
//
// Inputs:
//   j -          jet type in integer
//
// Returns:
//   the jet type in string
//
{


    switch (j) {

    case hitfit::isr_label:
        return std::string("g");
    case hitfit::lepb_label:
        return std::string("b");
    case hitfit::hadb_label:
        return std::string("B");
    case hitfit::hadw1_label:
        return std::string("W");
    case hitfit::hadw2_label:
        return std::string("W");
    case hitfit::higgs_label:
        return std::string("h");
    case hitfit::unknown_label:
        return std::string("?");
    default:
        return std::string("?");
    }

    return std::string("?"); 

}

} // namespace hitfit
