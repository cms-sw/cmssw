//
// $Id: Fit_Result.cc,v 1.1 2011/05/26 09:47:00 mseidel Exp $
//
// File: src/Fit_Result.cc
// Purpose: Hold the result from a single kinematic fit.
// Created: Jul, 2000, sss, based on run 1 mass analysis code.
//
// CMSSW File      : src/Fit_Result.cc
// Original Author : Scott Stuart Snyder <snyder@bnl.gov> for D0
// Imported to CMSSW by Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>
//

/**
    @file Fit_Result.cc

    @brief Hold the result of one kinematic fit.  See the documentation
    for the header file Fit_Result.h for details.

    @author Scott Stuart Snyder <snyder@bnl.gov>

    @par Creation date:
    Jul 2000.

    @par Modification History:
    Apr 2009: Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>:
    Imported to CMSSW.<br>
    Nov 2009: Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>:
    Added Doxygen tags for automatic generation of documentation.

    @par Terms of Usage:
    With consent from the original author (Scott Snyder).

 */

#include "TopQuarkAnalysis/TopHitFit/interface/Fit_Result.h"
#include <ostream>
#include <cmath>


using std::ostream;
using std::abs;


namespace hitfit {


Fit_Result::Fit_Result (double chisq,
                        const Lepjets_Event& ev,
                        const Column_Vector& pullx,
                        const Column_Vector& pully,
                        double umwhad,
                        double utmass,
                        double mt,
                        double sigmt)
//
// Purpose: Constructor.
//
// Inputs:
//   chisq -       The fit chisq.
//   ev -          The event kinematics.
//   pullx -       The pull quantities for the well-measured variables.
//   pully -       The pull quantities for the poorly-measured variables.
//   umwhad -      The hadronic W mass before the fit.
//   utmass -      The top quark mass before the fit.
//   mt -          The top quark mass after the fit.
//   sigmt -       The top quark mass uncertainty after the fit.
//
  : _chisq (chisq),
    _umwhad (umwhad),
    _utmass (utmass),
    _mt (mt),
    _sigmt (sigmt),
    _pullx (pullx),
    _pully (pully),
    _ev (ev)
{
}


double Fit_Result::chisq () const
//
// Purpose: Return the fit chisq.
//
// Returns:
//   Return the fit chisq.
//
{
  return _chisq;
}


double Fit_Result::umwhad () const
//
// Purpose: Return the hadronic W mass before the fit.
//
// Returns:
//   The hadronic W mass before the fit.
//
{
  return _umwhad;
}


double Fit_Result::utmass () const
//
// Purpose: Return the top mass before the fit.
//
// Returns:
//   The top mass before the fit.
//
{
  return _utmass;
}


double Fit_Result::mt () const
//
// Purpose: Return the top mass after the fit.
//
// Returns:
//   The top mass after the fit.
//
{
  return _mt;
}


double Fit_Result::sigmt () const
//
// Purpose: Return the top mass uncertainty after the fit.
//
// Returns:
//   The top mass uncertainty after the fit.
//
{
  return _sigmt;
}


const Column_Vector& Fit_Result::pullx () const
//
// Purpose: Return the pull quantities for the well-measured variables.
//
// Returns:
//   The pull quantities for the well-measured variables.
//
{
  return _pullx;
}


const Column_Vector& Fit_Result::pully () const
//
// Purpose: Return the pull quantities for the poorly-measured variables.
//
// Returns:
//   The pull quantities for the poorly-measured variables.
//
{
  return _pully;
}


std::vector<int> Fit_Result::jet_types () 
//
// Purpose: Returns the jet types of the fit
//
{
  return _ev.jet_types();
}

const Lepjets_Event& Fit_Result::ev () const
//
// Purpose: Return the event kinematic quantities after the fit.
//
// Returns:
//   The event kinematic quantities after the fit.
//
{
  return _ev;
}


/**
    @brief Sort fit results based on their \f$\chi^{2}\f$.

    @param a The first instance of Fit_Result to compare.

    @param b The second instance of Fit_Result to compare.

    @par Return:
    <b>TRUE</b> if the first instance has smaller absolute value of
    \f$\chi^{2}\f$ than b.<br>
    <b>FALSE</b> all other cases.
*/
bool operator< (const Fit_Result& a, const Fit_Result& b)
//
// Purpose: Compare two objects by chisq.
//          The use of abs() is to make sure that the -1000 flags
//          go to the end.
//
// Inputs:
//   a -           The first object to compare.
//   b -           The second object to compare.
//
// Returns:
//   The result of the comparison.
//
{
  return abs (a._chisq) < abs (b._chisq);
}


  /**
     @brief Output stream operator, print the content of this Fit_Result to
     an output stream.

     @param s The output stream to which to write.

     @param res The instance of Fit_Result to be printed.
   */
std::ostream& operator<< (std::ostream& s, const Fit_Result& res)
//
// Purpose: Print the object to S.
//
// Inputs:
//   s -           The stream to which to write.
//   res -         The object to write.
//
// Returns:
//   The stream S.
//
{
  s << "chisq: "  << res._chisq  << "\n";
  s << "umwhad: " << res._umwhad << "\n";
  s << "utmass: " << res._utmass << "\n";
  s << "mt: "     << res._mt     << "\n";
  s << "sigmt: "  << res._sigmt << "\n";
  s << res._ev;
  s << "pullx: " << res._pullx.T();
  s << "pully: " << res._pully.T();
  return s;
}



} // namespace hitfit
