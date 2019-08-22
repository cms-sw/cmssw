#ifndef RealQuadEquation_H
#define RealQuadEquation_H

#include <cmath>
#include "FWCore/Utilities/interface/Visibility.h"

/** A numericaly stable and as fast as can be quadratic equation solver.
 *  The equation has the form A*x^2 + B*x + C = 0
 */

struct dso_internal RealQuadEquation {
  double first;
  double second;
  bool hasSolution;

  RealQuadEquation(double A, double B, double C) {
    double D = B * B - 4 * A * C;
    if (D < 0)
      hasSolution = false;
    else {
      hasSolution = true;
      auto q = -0.5 * (B + std::copysign(std::sqrt(D), B));
      first = q / A;
      second = C / q;
    }
  }
};

#endif
