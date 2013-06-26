//
// $Id: Fit_Results.h,v 1.1 2011/05/26 09:46:53 mseidel Exp $
//
// File: hitfit/Fit_Results.h
// Purpose: Hold the results from kinematic fitting.
// Created: Jul, 2000, sss, based on run 1 mass analysis code.
//
// This is a set of lists of fit results.
// Each list corresponds to some subset of jet permutations:
//   all permutations, btagged permutations, etc.
//
// CMSSW File      : interface/Fit_Results.h
// Original Author : Scott Stuart Snyder <snyder@bnl.gov> for D0
// Imported to CMSSW by Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>
//

/**
    @file Fit_Results.h

    @brief Hold set(s) of results from
    more than one kinematic fits.  Each set correspond to some
    subset of jet permutation: all permutations, btagged permutations, etc.

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

#ifndef HITFIT_FIT_RESULTS_H
#define HITFIT_FIT_RESULTS_H


#include "TopQuarkAnalysis/TopHitFit/interface/Fit_Result_Vec.h"
#include "TopQuarkAnalysis/TopHitFit/interface/matutil.h"
#include <vector>
#include <iosfwd>


namespace hitfit {


class Lepjets_Event;


/**
    @class Fit_Results

    @brief Holds set(s) of results from more than one kinematic fits.
*/
class Fit_Results
//
// Purpose: Hold the results from kinematic fitting.
//
{
public:
  // Constructor.  Make N_LISTS lists, each of maximum length MAX_LEN.
  /**
     @brief Constructor, make <i>n_list</i> of lists, each of maximum
     length <i>max_len</i>.

     @param max_len The maximum length of each list.

     @param n_lists The number of lists.
   */
  Fit_Results (int max_len, int n_lists);

  // Return the Ith list.
  /**
     @brief Access the <i>i-</i>th list

     @param i The index of the list.
   */
  const Fit_Result_Vec& operator[] (std::vector<Fit_Result_Vec>::size_type i) const;

  // Add a new result.  LIST_FLAGS tells on which lists to enter it.
  /**
     @brief Add a new fit result.

     @param chisq The fit \f$\chi^{2}\f$.

     @param ev The event kinematics.

     @param pullx The pull quantities for the well-measured variables.

     @param pully The pull quantities for the poorly-measured variables.

     @param umwhad The hadronic W mass before fit.

     @param utmass The top quark mass before the fit.

     @param mt The top quark mass after the fit.

     @param sigmt The top quark mass uncertainty after the fit.

     @param list_flags Vector indicating to which lists the result should
     be added.  This vector should have a same length as the internal
     object _v.
   */
  void push (double chisq,
             const Lepjets_Event& ev,
             const Column_Vector& pullx,
             const Column_Vector& pully,
             double umwhad,
             double utmass,
             double mt,
             double sigmt,
             const std::vector<int>& list_flags);

  // Print this object.
  friend std::ostream& operator<< (std::ostream& s, const Fit_Results& res);


private:
  // The object state.
  std::vector<Fit_Result_Vec> _v;
};


} // namespace hitfit


#endif // not HITFIT_FIT_RESULT_H
