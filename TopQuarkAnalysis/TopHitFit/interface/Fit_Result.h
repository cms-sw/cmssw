//
// $Id: Fit_Result.h,v 1.1 2011/05/26 09:46:53 mseidel Exp $
//
// File: hitfit/Fit_Result.h
// Purpose: Hold the result from a single kinematic fit.
// Created: Jul, 2000, sss, based on run 1 mass analysis code.
//
// These objects are reference-counted.
//
// CMSSW File      : interface/Fit_Result.h
// Original Author : Scott Stuart Snyder <snyder@bnl.gov> for D0
// Imported to CMSSW by Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>
//

/**
    @file Fit_Result.h

    @brief Hold the result of one kinematic fit.

    @author Scott Stuart Snyder <snyder@bnl.gov>

	@par Creation date:
    July 2000.

    @par Modification History:
    Apr 2009: Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>:
    Imported to CMSSW.<br>
    Nov 2009: Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>:
    Added Doxygen tags for automatic generation of documentation.

    @par Terms of Usage:
    With consent from the original author (Scott Snyder).

 */
#ifndef HITFIT_FIT_RESULT_H
#define HITFIT_FIT_RESULT_H


#include "TopQuarkAnalysis/TopHitFit/interface/Refcount.h"
#include "TopQuarkAnalysis/TopHitFit/interface/Lepjets_Event.h"
#include "TopQuarkAnalysis/TopHitFit/interface/matutil.h"
#include <iosfwd>
#include <vector>

namespace hitfit {


/**
    @class Fit_Result

    @brief Hold the result of one kinematic fit.

 */
class Fit_Result
  : public hitfit::Refcount
//
// Purpose: Hold the result from a single kinematic fit.
//
{
public:
  // Constructor.  Provide the results of the fit.
  /**
     @brief Constructor, provide the results of the fit.

     @param chisq The  \f$ \chi^{2} \f$  of the fit.
     @param ev The event kinematics after the fit.
     @param pullx Pull quantities for the well-measured variables.
     @param pully Pull quantities for the poorly-measured variables.
     @param umwhad The hadronic  \f$ W- \f$ boson mass before the fit.
     @param utmass The top quark mass before the fit, the average
     of leptonic and hadronic top mass.
     @param mt The top quark mass after the fit.
     @param sigmt The top quark mass uncertainty after the fit.
   */
  Fit_Result (double chisq,
              const Lepjets_Event& ev,
              const Column_Vector& pullx,
              const Column_Vector& pully,
              double umwhad,
              double utmass,
              double mt,
              double sigmt);

  // Get back the fit result.
  /**
     @brief Return the  \f$ \chi^{2} \f$  of the fit.
   */
  double chisq () const;

  /**
     @brief Return the hadronic  \f$ W- \f$ boson mass before the fit.
   */
  double umwhad () const;

  /**
     @brief Return the top quark mass before the fit.
   */
  double utmass () const;

  /**
     @brief Return the top quark mass after the fit.
   */
  double mt () const;

  /**
     @brief Return the top quark mass uncertainty after the fit.
   */
  double sigmt () const;

  /**
     @brief Return the pull quantities for the well-measured variables.
   */
  const Column_Vector& pullx () const;

  /**
     @brief Return the pull quantities for the poorly-measured variables.
   */
  const Column_Vector& pully () const;

  /**
     @brief Return the event kinematics quantities after the fit.
   */
  const Lepjets_Event& ev () const;

  // For sorting by chisq.
  friend bool operator< (const Fit_Result& a, const Fit_Result& b);

  // Print this object.
  friend std::ostream& operator<< (std::ostream& s, const Fit_Result& res);

  // Get the jet-types permutation
  /**
     @brief Return the list of jet types for this event.

   */
  std::vector<int> jet_types();

private:
  // Store the results of the kinematic fit.

  /**
     The  \f$ \chi^{2} \f$  of the fit.
   */
  double _chisq;

  /**
     The hadronic  \f$ W- \f$ boson mass before the fit.
   */
  double _umwhad;

  /**
     The top quark mass before the fit.
   */
  double _utmass;

  /**
     The top quark mass after the fit.
   */
  double _mt;

  /**
     The top quark mass uncertainty after the fit.
   */
  double _sigmt;

  /**
     Pull quantities for the well-measured variables.
   */
  Column_Vector _pullx;

  /**
     Pull quantities for the poorly-measured variables.
   */
  Column_Vector _pully;

  /**
     The event kinematics after the fit.
   */
  Lepjets_Event _ev;
};


} // namespace hitfit


#endif // not HITFIT_FIT_RESULT_H
