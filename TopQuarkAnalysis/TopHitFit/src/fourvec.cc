//
// $Id: fourvec.cc,v 1.1 2011/05/26 09:47:00 mseidel Exp $
//
// File: src/fourvec.cc
// Purpose: Define 3- and 4-vector types for the hitfit package, and
//          supply a few additional operations.
// Created: Jul, 2000, sss, based on run 1 mass analysis code.
//
// CMSSW File      : src/fourvec.cc
// Original Author : Scott Stuart Snyder <snyder@bnl.gov> for D0
// Imported to CMSSW by Haryo Sumowidagdo <Suharyo.Sumowidagdo@cern.ch>
//

/**
    @file fourvec.cc

    @brief Define three-vector and four-vector classes for the HitFit
    package, and supply a few additional operations. See the documentation
    for the header file fourvec.h for details.

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

#include "TopQuarkAnalysis/TopHitFit/interface/fourvec.h"
#include <cmath>

using std::fabs;
using std::sqrt;
using std::sin;
using std::cos;
using std::atan;
using std::exp;
using std::log;
using std::tan;
using std::atan2;


namespace { // unnamed namespace


double cal_th (double theta, double z)
//
// Purpose: Helper for deteta.  Find `calorimeter theta' given
//          physics theta and z-vertex.  Copied from run 1 code.
//
// Inputs:
//   theta -       Physics theta.
//   z -           Z-vertex.
//
// Returns:
//   Calorimeter theta.
//
{
  const double R_CC = 91.6;
  const double Z_EC = 178.9;
  const double BIGG = 1e8;

  double tanth;
  if (fabs (cos (theta)) < 1/BIGG)
    tanth = BIGG * sin (theta);
  else
    tanth = tan (theta);

  double z_cc = R_CC / tanth + z;

  if (fabs (z_cc) < Z_EC)
    theta = atan2 (R_CC, z_cc);

  else {
    double zz = Z_EC;
    if (tanth < 0) zz = - zz;
    double r_ec = (zz-z) * tanth;
    theta = atan2 (r_ec, zz);
  }

  if (theta < 0) theta += 2 * M_PI;
  return theta;
}


} // unnamed namespace


namespace hitfit {


void adjust_p_for_mass (Fourvec& v, double mass)
//
// Purpose: Adjust the 3-vector part of V (leaving the energy unchanged)
//          so that it has mass MASS.  (Provided that is possible.)
//
// Inputs:
//   v -           The 4-vector to scale.
//   mass -        The desired mass of the 4-vector.
//
// Outputs:
//   v -           The scaled 4-vector.
//
{
  CLHEP::Hep3Vector vect = v.vect();
  double old_p2 = vect.mag2();
  if (old_p2 == 0)
    return;
  double new_p2 = v.e()*v.e() - mass*mass;
  if (new_p2 < 0)
    new_p2 = 0;
  vect *= sqrt (new_p2 / old_p2);
  v.setVect (vect);
}


void adjust_e_for_mass (Fourvec& v, double mass)
//
// Purpose: Adjust the energy component of V (leaving the 3-vector part
//          unchanged) so that it has mass MASS.
//
// Inputs:
//   v -           The 4-vector to scale.
//   mass -        The desired mass of the 4-vector.
//
// Outputs:
//   v -           The scaled 4-vector.
//
{
  v.setE (sqrt (v.vect().mag2() + mass*mass));
}


void rottheta (Fourvec& v, double theta)
//
// Purpose: Rotate V through polar angle THETA.
//
// Inputs:
//   v -           The 4-vector to rotate.
//   theta -       The rotation angle.
//
// Outputs:
//   v -           The rotated 4-vector.
//
{
  double s = sin (theta), c = cos (theta);
  double old_pt = v.perp();
  double new_pt =  old_pt*c - v.z()*s;
  v.setZ (old_pt*s + v.z()*c);

  v.setX (v.x() * new_pt / old_pt);
  v.setY (v.y() * new_pt / old_pt);
}


void roteta (Fourvec& v, double eta)
//
// Purpose: Rotate a Fourvec through a polar angle such that
//          its pseudorapidity changes by ETA.
//
// Inputs:
//   v -           The 4-vector to rotate.
//   eta -         The rotation angle.
//
// Outputs:
//   v -           The rotated 4-vector.
//
{
  double theta1 = v.theta ();
  double eta1 = theta_to_eta (theta1);
  double eta2 = eta1 + eta;
  double theta2 = eta_to_theta (eta2);

  rottheta (v, theta1 - theta2);
}


double eta_to_theta (double eta)
//
// Purpose: Convert psuedorapidity to polar angle.
//
// Inputs:
//   eta -         Pseudorapidity.
//
// Returns:
//   Polar angle.
//
{
  return 2 * atan (exp (-eta));
}


double theta_to_eta (double theta)
//
// Purpose: Convert polar angle to psuedorapidity.
//
// Inputs:
//   theta -       Polar angle.
//
// Returns:
//   Pseudorapidity.
//
{
  return - log (tan (theta / 2));
}


double deteta (const Fourvec& v, double zvert)
//
// Purpose: Get the detector eta (D0-specific).
//
// Inputs:
//   v -           Vector on which to operate.
//   zvert -       Z-vertex.
//
// Returns:
//   Detector eta of V.
//
{
  return theta_to_eta (cal_th (v.theta(), zvert));
}


double phidiff (double phi)
//
// Purpose: Handle wraparound for a difference in azimuthal angles.
//
// Inputs:
//   phi -         Azimuthal angle.
//
// Returns:
//   PHI normalized to the range -pi .. pi.
//
{
  while (phi < -M_PI)
    phi += 2 * M_PI;
  while (phi > M_PI)
    phi -= 2*M_PI;
  return phi;
}


double delta_r (const Fourvec& a, const Fourvec& b)
//
// Purpose: Find the distance in R between two four-vectors.
//
// Inputs:
//   a -           First four-vector.
//   b -           Second four-vector.
//
// Returns:
//   the distance in R between A and B.
//
{
  double deta = a.pseudoRapidity() - b.pseudoRapidity();
  double dphi = phidiff (a.phi() - b.phi());
  return sqrt (deta*deta + dphi*dphi);
}


} // namespace hitfit
