//
// $Id: Top_Decaykin.h,v 1.2 2013/05/28 17:55:59 gartung Exp $
//
// File: hitfit/Top_Decaykin.h
// Purpose: Calculate some kinematic quantities for ttbar events.
// Created: Jul, 2000, sss, based on run 1 mass analysis code.
//
// CMSSW File      : interface/Top_Decaykin.h
// Original Author : Scott Stuart Snyder <snyder@bnl.gov> for D0
// Imported to CMSSW by Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>
//


/**
    @file Top_Decaykin.h

    @brief A class to hold functions to calculate kinematic quantities
    of interest in  \f$ t\bar{t} \to \ell + 4 \mathrm{jets} \f$  events.

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

#ifndef HITFIT_TOP_DECAYKIN_H
#define HITFIT_TOP_DECAYKIN_H

#include "CLHEP/Vector/Boost.h"

#include "TopQuarkAnalysis/TopHitFit/interface/fourvec.h"
#include <iosfwd>


namespace hitfit {


class Lepjets_Event;


/**
    @brief A class to hold functions to calculate kinematic quantities
    of interest in  \f$ t\bar{t} \to \ell + 4 \mathrm{jets} \f$  events.
    This class has no state, only static member functions.
 */
class Top_Decaykin
//
// Purpose: Calculate some kinematic quantities for ttbar events.
//          This class has no state --- just static member functions.
//
{
public:
  // Solve for the longitudinal z-momentum that makes the leptonic
  // top have mass TMASS.
  /**
     @brief Solve for the neutrino longitudinal  \f$ z- \f$ momentum
     that makes the leptonic top have a certain value of mass.  Returns
     <b>TRUE</b> if there were real solutions.  Returns <b>FALSE</b> if there
     were only complex solutions.  In case of complex solutions, the real
     components of the solutions are given.

     @param ev Input:The event to solve.

     @param tmass Input: The desired value of top quark  mass in GeV.

     @param nuz1 Output: The first solution (smaller absolute value).

     @param nuz2 Output: The second solution.
   */
  static bool solve_nu_tmass (const Lepjets_Event& ev, double tmass,
                              double& nuz1, double& nuz2);

  // Solve for the longitudinal z-momentum that makes the leptonic
  // top have mass TMASS, with imaginary component returned.
  /**
     @brief Solve for the neutrino longitudinal  \f$ z- \f$ momentum
     that makes the leptonic top have a certain value of mass.  The
     complex component of the solutions are also given.
     Returns <b>TRUE</b> if there were real solutions.
     Returns <b>FALSE</b> if there were only complex solutions.
     In case of real solutions, the first solution is the one which have
     smaller absolute value.  In case of imaginary solutions (which are
     complex conjugate of each other), the first solution is the one
     which have imaginary component in the lower half of the complex plane,
     i.e., the one which have negative imaginary component).

     @param ev Input:The event to solve.

     @param tmass Input: The desired value of top quark  mass in GeV.

     @param re_nuz1 Output: Real component of the first solution.

     @param im_nuz1 Output: Imaginary component of the first solution.

     @param re_nuz2 Output: Real component of the second solution.

     @param im_nuz2 Output: Imaginary component of the second solution.

   */
  static bool solve_nu_tmass (const Lepjets_Event& ev, double tmass,
                              double& re_nuz1, double& im_nuz1,
                              double& re_nuz2, double& im_nuz2);

  // Solve for the longitudinal z-momentum that makes the leptonic
  // W have mass WMASS.
  /**
     @brief Solve for the longitudinal  \f$ z- \f$ momentum that makes the
     leptonic  \f$ W \f$ -boson to have a certain value of mass.  Returns
     <b>TRUE</b> if there were real solutions.  Returns <b>FALSE</b> if there
     were only complex solutions.  In case of complex solutions, the real
     components of the solutions are given.


     @param ev Input: The event to solve.

     @param wmass Input: The desired mass of the  \f$ W- \f$ boson in GeV.

     @param nuz1 Output: First solution (smaller absolute value).

     @param nuz2 Output: Second solution.

   */
  static bool solve_nu (const Lepjets_Event& ev, double wmass,
                        double& nuz1, double& nuz2);

  // Solve for the longitudinal z-momentum that makes the leptonic
  // W have mass WMASS, with imaginary component returned.
  /**
     @brief Solve for the longitudinal  \f$ z- \f$ momentum that makes the
     leptonic  \f$ W \f$ -boson to have a certain value of mass.  The
     complex component of the solutions are also given.
     Returns <b>TRUE</b> if there were real solutions.
     Returns <b>FALSE</b> if there were only complex solutions.
     In case of real solutions, the first solution is the one which have
     smaller absolute value.  In case of imaginary solutions (which are
     complex conjugate of each other), the first solution is the one
     which have imaginary component in the lower half of the complex plane,
     i.e., the one which have negative imaginary component).

     @param ev Input: The event to solve.

     @param wmass Input: The desired mass of the  \f$ W- \f$ boson in GeV.

     @param re_nuz1 Output: Real component of the first solution.

     @param im_nuz1 Output: Imaginary component of the first solution.

     @param re_nuz2 Output: Real component of the second solution.

     @param im_nuz2 Output: Imaginary component of the second solution.

   */
  static bool solve_nu (const Lepjets_Event& ev, double wmass,
                        double& re_nuz1, double& im_nuz1,
                        double& re_nuz2, double& im_nuz2);

  // Sum up the appropriate 4-vectors to find the hadronic W.
  /**
     @brief Sum up the appropriate four-momenta to find the hadronic
      \f$ W- \f$ boson.

     @param ev The event.
   */
  static Fourvec hadw (const Lepjets_Event& ev);

  // Find the higher pT jet from hadronic W
  /**
     @brief Return the hadronic  \f$ W- \f$ boson jet which have higher
      \f$ p_{T} \f$ .

     @param ev The event.
   */
  static Fourvec hadw1 (const Lepjets_Event& ev);

  // Find the lower pT jet from hadronic W
  /**
     @brief Return the hadronic  \f$ W- \f$ boson jet which have lower
      \f$ p_{T} \f$ .

     @param ev The event.
   */
  static Fourvec hadw2 (const Lepjets_Event& ev);

  // Sum up the appropriate 4-vectors to find the leptonic W.
  /**
     @brief Sum up the appropriate four-momenta to find the leptonic
      \f$ W- \f$ boson.

     @param ev The event.
   */
  static Fourvec lepw (const Lepjets_Event& ev);

  // Sum up the appropriate 4-vectors to find the hadronic t.
  /**
     @brief Sum up the appropriate four-momenta to find the hadronic
     top quark.

     @param ev The event.
   */
  static Fourvec hadt (const Lepjets_Event& ev);

  // Sum up the appropriate 4-vectors to find the leptonic t.
  /**
     @brief Sum up the appropriate four-momenta to find the leptonic
     top quark.

     @param ev The event.
   */
  static Fourvec lept (const Lepjets_Event& ev);

  // Print kinematic information for EV.
  /**
     @brief Print the kinematic information for an event.

     @param s The stream of which to write.

     @param ev The event to be printed.
   */
  static std::ostream& dump_ev (std::ostream& s, const Lepjets_Event& ev);

  // Solve cos theta star
  /**
     @brief Calculate  \f$ \cos \theta^{*} \f$  in top quark decay.

     @param fermion The four-momentum of fermion from  \f$ W- \f$ boson
     from top decay.

     @param W The four-momentum of  \f$ W \f$ boson from top decay.

     @param top The four-momentum of top.
   */
  static double cos_theta_star(const Fourvec& fermion,
                               const Fourvec& W,
                               const Fourvec& top);

  // Solve cos theta star in lepton side of lepton+jets event
  /**
     @brief Calculate the lepton  \f$ \cos \theta^{*} \f$  in top
     quark leptonic decay.

     @param ev The event to solve.
   */
  static double cos_theta_star(const Lepjets_Event& ev);

  // Solve cos theta star in lepton side of lepton+jets event
  /**
     @brief Calculate the lepton  \f$ \cos \theta^{*} \f$  in top
     quark leptonic decay.

     @param ev The event to solve.
   */
  static double cos_theta_star_lept(const Lepjets_Event& ev);

  // Solve cos theta star in hadronic side of lepton+jets event
  /**
     @brief Calculate the hadronic  \f$ \cos \theta^{*} \f$  in top
     quark leptonic decay.  As there is no information on the weak
     isospin component of the fermion, the absolute value of
      \f$ \cos \theta^{*} \f$  will be returned (the solutions for
     up-type and down-type fermions will differ only in sign but not
     in magnitude).

     @param ev The event to solve.
   */
  static double cos_theta_star_hadt(const Lepjets_Event& ev);


};


} // namespace hitfit


#endif // not HITFIT_TOP_DECAYKIN_H

