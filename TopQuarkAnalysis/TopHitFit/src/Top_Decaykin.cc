//
// $Id: Top_Decaykin.cc,v 1.2 2013/05/28 17:55:59 gartung Exp $
//
// File: src/Top_Decaykin.cc
// Purpose: Calculate some kinematic quantities for ttbar events.
// Created: Jul, 2000, sss, based on run 1 mass analysis code.
//
// CMSSW File      : src/Top_Decaykin.cc
// Original Author : Scott Stuart Snyder <snyder@bnl.gov> for D0
// Imported to CMSSW by Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>
//


/**
    @file Top_Decaykin.cc

    @brief A class to hold functions to calculate kinematic quantities
    of interest in \f$t\bar{t} \to \ell + 4 \mathrm{jets}\f$ events.
    See the documentation for the header file Top_Decaykin.h for details.

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

#include "TopQuarkAnalysis/TopHitFit/interface/Top_Decaykin.h"
#include "TopQuarkAnalysis/TopHitFit/interface/Lepjets_Event.h"
#include "TopQuarkAnalysis/TopHitFit/interface/fourvec.h"
#include <cmath>
#include <algorithm>
#include <ostream>


using std::sqrt;
using std::abs;
using std::swap;
using std::ostream;


namespace hitfit {


namespace {


/**
    @brief Sum the four-momenta of all leptons in an event.

    @param ev The event.
 */
Fourvec leptons (const Lepjets_Event& ev)
//
// Purpose: Sum all leptons in EV.
//
// Inputs:
//   ev -          The event.
//
// Returns:
//   The sum of all leptons in EV.
//
{
  return (ev.sum (lepton_label) +
          ev.sum (electron_label) +
          ev.sum (muon_label));
}


} // unnamed namespace


bool Top_Decaykin::solve_nu_tmass (const Lepjets_Event& ev,
                                   double tmass,
                                   double& nuz1, double& nuz2)
//
// Purpose: Solve for the neutrino longitudinal z-momentum that makes
//          the leptonic top have mass TMASS.
//
// Inputs:
//   ev -          The event to solve.
//   tmass -       The desired top mass.
//
// Outputs:
//   nuz1 -        First solution (smaller absolute value).
//   nuz2 -        Second solution.
//
// Returns:
//   True if there was a real solution.  False if there were only
//   imaginary solutions.  (In that case, we just set the imaginary
//   part to zero.)
//
{
  bool discrim_flag = true;

  const Fourvec& vnu = ev.met ();
  Fourvec cprime = leptons (ev) + ev.sum (lepb_label);
  double alpha1 = tmass*tmass - cprime.m2();
  double a = 2 * 4 * (cprime.z()*cprime.z() - cprime.e()*cprime.e());
  double alpha = alpha1 + 2*(cprime.x()*vnu.x() + cprime.y()*vnu.y());
  double b = 4 * alpha * cprime.z();
  double c = alpha*alpha - 4 * cprime.e()*cprime.e() * vnu.vect().perp2();
  double d = b*b - 2*a*c;
  if (d < 0) {
    discrim_flag = false;
    d = 0;
  }

  double dd = sqrt (d);
  nuz1 = (-b + dd)/a;
  nuz2 = (-b - dd)/a;
  if (abs (nuz1) > abs (nuz2))
    swap (nuz1, nuz2);

  return discrim_flag;
}


bool Top_Decaykin::solve_nu_tmass (const Lepjets_Event& ev,
                                   double tmass,
                                   double& re_nuz1,
                                   double& im_nuz1,
                                   double& re_nuz2,
                                   double& im_nuz2)
//
// Purpose: Solve for the neutrino longitudinal z-momentum that makes
//          the leptonic top have mass TMASS, including the imaginary
//          component of the z-component of the neutrino'somentum.
//
// Inputs:
//   ev -          The event to solve.
//   tmass -       The desired top mass.
//
// Outputs:
//   re_nuz1 -     Real component of the first solution.
//   im_nuz1 -     Imaginary component of the first solution (in the lower half of
//                 the complex plane).
//   re_nuz2 -     Real component of the second solution.
//   im_nuz2 -     Imaginary component of the second solution (in the upper half of
//                 the complex plane).
//
// Returns:
//   True if there was a real solution.  False if there were only
//   complex solutions.
//
{
  bool discrim_flag = true;

  const Fourvec& vnu = ev.met ();
  Fourvec cprime = leptons (ev) + ev.sum (lepb_label);
  double alpha1 = tmass*tmass - cprime.m2();
  double a = 2 * 4 * (cprime.z()*cprime.z() - cprime.e()*cprime.e());
  // Haryo's note: Here a is equivalent to '2a' in the quadratic
  // equation ax^2 + bx + c = 0
  double alpha = alpha1 + 2*(cprime.x()*vnu.x() + cprime.y()*vnu.y());
  double b = 4 * alpha * cprime.z();
  double c = alpha*alpha - 4 * cprime.e()*cprime.e() * vnu.vect().perp2();
  double d = b*b - 2*a*c;
  if (d < 0) {
    discrim_flag = false;
  }

  if (discrim_flag) {

      re_nuz1 = (-b + sqrt(d))/a ;
      im_nuz1 = 0 ;
      re_nuz2 = (-b - sqrt(d))/a ;
      im_nuz2 = 0 ;
      if (abs(re_nuz1) > abs(re_nuz2)) {
          swap(re_nuz1,re_nuz2);
      }

  } else {

      // Take absolute value of the imaginary component of nuz, in case
      // a is negative, before multiplying by +1 or -1 to get the upper-half
      // or lower-half imaginary value.
      re_nuz1   = -b / a;
      im_nuz1   = -fabs(sqrt(-d)/a );
      re_nuz2   = -b / a;
      im_nuz2   =  fabs(sqrt(-d)/a );


  }


  return discrim_flag;
}


bool Top_Decaykin::solve_nu (const Lepjets_Event& ev,
                             double wmass,
                             double& nuz1,
                             double& nuz2)
//
// Purpose: Solve for the neutrino longitudinal z-momentum that makes
//          the leptonic W have mass WMASS.
//
// Inputs:
//   ev -          The event to solve.
//   wmass -       The desired W mass.
//
// Outputs:
//   nuz1 -        First solution (smaller absolute value).
//   nuz2 -        Second solution.
//
// Returns:
//   True if there was a real solution.  False if there were only
//   imaginary solutions.  (In that case, we just set the imaginary
//   part to zero.)
//
{
  bool discrim_flag = true;

  Fourvec vnu  = ev.met();
  Fourvec vlep = leptons (ev);

  double x = vlep.x()*vnu.x() + vlep.y()*vnu.y() + wmass*wmass/2;
  double a = vlep.z()*vlep.z() - vlep.e()*vlep.e();
  double b = 2*x*vlep.z();
  double c = x*x - vnu.perp2() * vlep.e()*vlep.e();

  double d = b*b - 4*a*c;
  if (d < 0) {
    d = 0;
    discrim_flag = false;
  }

  nuz1 = (-b + sqrt (d))/2/a;
  nuz2 = (-b - sqrt (d))/2/a;
  if (abs (nuz1) > abs (nuz2))
    swap (nuz1, nuz2);

  return discrim_flag;
}


bool Top_Decaykin::solve_nu (const Lepjets_Event& ev,
                             double wmass,
                             double& re_nuz1,
                             double& im_nuz1,
                             double& re_nuz2,
                             double& im_nuz2)
//
// Purpose: Solve for the neutrino longitudinal z-momentum that makes
//          the leptonic W have mass WMASS, including the imaginary
//          component of the z-component of the neutrino'somentum.
//
// Inputs:
//   ev -          The event to solve.
//   wmass -       The desired W mass.
//
// Outputs:
//   re_nuz1 -     Real component of the first solution.
//   im_nuz1 -     Imaginary component of the first solution  (in the lower half of
//                 the complex plane).
//   re_nuz2 -     Real component of the second solution.
//   im_nuz2 -     Imaginary component of the second solution  (in the upper half of
//                 the complex plane).
//
// Returns:
//   True if there was a real solution.  False if there were only
//   complex solutions.
//x
{
  bool discrim_flag = true;

  Fourvec vnu  = ev.met();
  Fourvec vlep = leptons (ev);

  double x = vlep.x()*vnu.x() + vlep.y()*vnu.y() + wmass*wmass/2;
  double a = vlep.z()*vlep.z() - vlep.e()*vlep.e();
  double b = 2*x*vlep.z();
  double c = x*x - vnu.perp2() * vlep.e()*vlep.e();

  double d = b*b - 4*a*c;
  if (d < 0) {
    discrim_flag = false;
  }

  if (discrim_flag) {

      re_nuz1 = (-b + sqrt(d))/2/a ;
      im_nuz1 = 0.0 ;
      re_nuz2 = (-b - sqrt(d))/2/a ;
      im_nuz2 = 0.0 ;
      if (fabs(re_nuz1) > fabs(re_nuz2)) {
          swap(re_nuz1,re_nuz2);
      }

  } else {

      // Take absolute value of the imaginary component of nuz, in case
      // a is negative, before multiplying by +1 or -1 to get the upper-half
      // or lower-half imaginary value.

      re_nuz1 = -b /2/a ;
      im_nuz1 = -fabs(sqrt(-d)/a);
      re_nuz2 = -b /2/a ;
      im_nuz2 =  fabs(sqrt(-d)/a);

  }

  return discrim_flag;
}


Fourvec Top_Decaykin::hadw (const Lepjets_Event& ev)
//
// Purpose: Sum up the appropriate 4-vectors to find the hadronic W.
//
// Inputs:
//   ev -          The event.
//
// Returns:
//   The hadronic W.
//
{
  return (ev.sum (hadw1_label) + ev.sum (hadw2_label));
}


Fourvec Top_Decaykin::hadw1 (const Lepjets_Event& ev)
//
// Purpose:
//
// Inputs:
//   ev -          The event.
//
// Returns:
//   The higher-pT hadronic jet from W
//
{
  return ev.sum (hadw1_label);
}


Fourvec Top_Decaykin::hadw2 (const Lepjets_Event& ev)
//
// Purpose:
//
// Inputs:
//   ev -          The event.
//
// Returns:
//   The lower-pT hadronic jet from W
//
//
{
  return ev.sum (hadw2_label);
}


Fourvec Top_Decaykin::lepw (const Lepjets_Event& ev)
//
// Purpose: Sum up the appropriate 4-vectors to find the leptonic W.
//
// Inputs:
//   ev -          The event.
//
// Returns:
//   The leptonic W.
//
{
  return (leptons (ev) + ev.met ());
}


Fourvec Top_Decaykin::hadt (const Lepjets_Event& ev)
//
// Purpose: Sum up the appropriate 4-vectors to find the hadronic t.
//
// Inputs:
//   ev -          The event.
//
// Returns:
//   The hadronic t.
//
{
  return (ev.sum (hadb_label) + hadw (ev));
}


Fourvec Top_Decaykin::lept (const Lepjets_Event& ev)
//
// Purpose: Sum up the appropriate 4-vectors to find the leptonic t.
//
// Inputs:
//   ev -          The event.
//
// Returns:
//   The leptonic t.
//
{
  return (ev.sum (lepb_label) + lepw (ev));
}


ostream& Top_Decaykin::dump_ev (std::ostream& s, const Lepjets_Event& ev)
//
// Purpose: Print kinematic information for EV.
//
// Inputs:
//   s -           The stream to which to write.
//   ev -          The event to dump.
//
// Returns:
//   The stream S.
//
{
  s << ev;
  Fourvec p;

  p = lepw (ev);
  s << "lepw " << p << " " << p.m() << "\n";
  p = lept (ev);
  s << "lept " << p << " " << p.m() << "\n";
  p = hadw (ev);
  s << "hadw " << p << " " << p.m() << "\n";
  p = hadt (ev);
  s << "hadt " << p << " " << p.m() << "\n";

  return s;
}


double Top_Decaykin::cos_theta_star(const Fourvec& fermion,
                                    const Fourvec& W,
                                    const Fourvec& top)
//
// Purpose: Calculate cos theta star in top decay
//
// Inputs:
//   fermion -     The four momentum of fermion from W
//   W -           The four momentum of W from top
//   top -         The four momentum of top
// Returns:
//   cos theta star
//
{

    if (W.isLightlike() || W.isSpacelike()) {
        return 100.0;
    }

    CLHEP::HepBoost BoostWCM(W.findBoostToCM());

    CLHEP::Hep3Vector boost_v3fermion       = BoostWCM(fermion).vect();
    CLHEP::Hep3Vector boost_v3top           = BoostWCM(top).vect();

    double costhetastar = boost_v3fermion.cosTheta(-boost_v3top);

    return costhetastar;
}


double Top_Decaykin::cos_theta_star(const Lepjets_Event& ev)
//
// Purpose: Calculate lepton cos theta star in top decay
//
// Inputs:
//   ev -          A lepton+jets event
// Returns:
//   cos theta star of lepton
//
{

    return cos_theta_star(leptons(ev),
                          lepw(ev),
                          lept(ev));

}


double Top_Decaykin::cos_theta_star_lept(const Lepjets_Event& ev)
//
// Purpose: Calculate lepton cos theta star in top decay
//
// Inputs:
//   ev -          A lepton+jets event
// Returns:
//   cos theta star of lepton
//
{

    return cos_theta_star(ev);

}


double Top_Decaykin::cos_theta_star_hadt(const Lepjets_Event& ev)
//
// Purpose: Calculate absolute value of cos theta star of
//          one of the hadronic W jet from hadronic top.
//
// Inputs:
//   ev -          A lepton+jets event
// Returns:
//   absolute value of cos theta star
//
{

    return fabs(cos_theta_star(hadw1(ev),
                               hadw(ev),
                               hadt(ev)));

}


} // namespace hitfit


