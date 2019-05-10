
/*
 * This class collects code fome the pakage MBDigitizer, which interfaces
 * the Rovelli-Gresele parametrization used with ORCA 6. It is included here
 * for comparison with the current parametrization.
 *
 */

#include "Utils.h"

#include "DTAngleParam.h"
#include "DTBNormParam.h"

#include <cmath>
#include <iostream>

// cfr. DTTimeDrift::ShortDistCorrect
float ShortDistCorrect(float x) {
  float mu_drift_corr = fabs(x);  // cfr. DTSpaceDrift:69
  float min_dist = 0.03;

  const float factor_range = 2.0;
  float short_range = factor_range * min_dist;
  float punch_dist = short_range - mu_drift_corr;
  if (punch_dist > 0.0)
    mu_drift_corr = short_range - punch_dist * (factor_range - 1.0) / factor_range;

  return mu_drift_corr;
}

// x in cm; B in T
// Bwire = By in ORCA r.f.
// Bnorm = Bz in ORCA r.f.

float oldParametrization(float x, float theta, float Bwire, float Bnorm) {
  // Change coordinates.
  Bwire = -Bwire;  // correct BUG in ORCA.

  //**********************************************************************

  // CFR DTTimeDrift::ShortDistCorrect
  x = ShortDistCorrect(x);

  // CFR DTTimeDrift::TimeCompute

  // set some cell parameters
  const float cell_width = 2.1;  // (mu_wire->wireType()->width())/2;

  const float volume_lim = cell_width - 0.0025;
  const float volume_par = cell_width - 0.1;
  const float edge_width = volume_lim - volume_par;

  // define coordinate with 0 inside the I beam
  // mu_drift_corr = x
  float xcoor = cell_width - fabs(x);

  if (xcoor < 0.0)
    xcoor = 0.0;

  if (xcoor > volume_lim) {
    return 0.0;
  }

  xcoor *= (2 / cell_width);

  // ****** patch to use the old parametrization with the new geometry ******
  //   bool extrapolate = ( mu_drift_corr > cell_width );
  //  float mu_drift_extr = ( extrapolate ? cell_width : mu_drift_corr );
  // ****** ********************************************************** ******
  float angle = theta;
  while (angle > 360.0)
    angle -= 360.0;
  if ((angle > 90.0) && (angle <= 270.0))
    angle -= 180.0;
  else if ((angle > 270.0) && (angle <= 360.0))
    angle -= 360.0;

  // access to CMS Magnetic Field
  if (Bwire < 0.0) {
    Bwire = fabs(Bwire);
    angle = -angle;
  }

  // handle cell edge cases
  bool out_of_volume = (xcoor > volume_par);
  float xpos = xcoor;
  if (out_of_volume) {
    xpos = xcoor;
    xcoor = volume_par;
  }

  float mu_time_drift = 0.;

  // call parametrization for the actual angle
  DTAngleParam angle_func(angle);
  mu_time_drift = angle_func.time(Bwire, xcoor);

  //  add correction for norma B field
  //@@ magnetic field not yet available
  DTBNormParam bnorm_func(Bnorm);
  mu_time_drift += bnorm_func.tcor(xcoor);

  // correct for cell edge cases
  if (out_of_volume)
    mu_time_drift *= (1.0 - ((xpos - volume_par) / edge_width));

  // correzione lunghezza cella
  mu_time_drift *= cell_width / 2.;

  mu_time_drift *= 1000.0;
  return mu_time_drift;
}
