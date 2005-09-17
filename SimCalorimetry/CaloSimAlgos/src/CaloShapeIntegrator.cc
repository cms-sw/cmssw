#include "SimCalorimetry/CaloSimAlgos/interface/CaloShapeIntegrator.h"

namespace cms {
  double CaloShapeIntegrator::operator() (double startTime) const {
    double sum = 0.;
     // not sure what the half-a-bin is for
    double time = startTime + 0.5;
    for(unsigned istep = 0; istep < BUNCHSPACE; ++istep) {
      sum += (*theShape)(time);
      ++time;
    }
    return sum;
  }
}

