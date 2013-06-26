//
// $Id: Fit_Results.cc,v 1.1 2011/05/26 09:47:00 mseidel Exp $
//
// File: src/Fit_Results.cc
// Purpose: Hold the results from kinematic fitting.
// Created: Jul, 2000, sss, based on run 1 mass analysis code.
//
// CMSSW File      : src/Fit_Results.cc
// Original Author : Scott Stuart Snyder <snyder@bnl.gov> for D0
// Imported to CMSSW by Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>
//


#include "TopQuarkAnalysis/TopHitFit/interface/Fit_Results.h"
#include "TopQuarkAnalysis/TopHitFit/interface/Fit_Result.h"
#include <ostream>
#include <cassert>


/**
    @file Fit_Results.cc

    @brief Holds set(s) of results from
    more than one kinematic fits.  See the documentation for the header file
    Fit_Results.h for details.

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

using std::ostream;


namespace hitfit {


Fit_Results::Fit_Results (int max_len, int n_lists)
//
// Purpose: Constructor.
//
// Inputs:
//   max_len -     The maximum length of each list.
//   n_lists -     The number of lists.
//
  : _v (n_lists, Fit_Result_Vec (max_len))
{
}


const Fit_Result_Vec& Fit_Results::operator[] (std::vector<Fit_Result_Vec>::size_type i) const
{
  assert (i < _v.size());
  return _v[i];
}


void Fit_Results::push (double chisq,
                        const Lepjets_Event& ev,
                        const Column_Vector& pullx,
                        const Column_Vector& pully,
                        double umwhad,
                        double utmass,
                        double mt,
                        double sigmt,
                        const std::vector<int>& list_flags)
//
// Purpose: Add a new fit result.
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
//   list_flags -  Vector indicating to which lists the result should
//                 be added.
//                 This vector should have a length of N_SIZE.
//                 Each element should be either 0 or 1, depending
//                 on whether or not the result should be added
//                 to the corresponding list.
//
{
  assert (list_flags.size() == _v.size());

  Fit_Result* res = new Fit_Result (chisq, ev, pullx, pully,
                                    umwhad, utmass, mt, sigmt);
  res->incref ();
  for (std::vector<Fit_Result_Vec>::size_type i=0; i < _v.size(); i++) {
    if (list_flags[i])
      _v[i].push (res);
  }
  res->decref ();
}


  /**
     @brief Output stream operator, print the content of this Fit_Results to
     an output stream.

     @param s The output stream to which to write.

     @param res The instance Fit_Results to be printed.
   */
std::ostream& operator<< (std::ostream& s, const Fit_Results& res)
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
  for (std::vector<Fit_Result_Vec>::size_type i=0; i < res._v.size(); i++)
    s << "List " << i << "\n" << res._v[i];
  return s;
}


} // namespace hitfit
