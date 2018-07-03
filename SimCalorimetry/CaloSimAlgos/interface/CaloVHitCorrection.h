#ifndef CaloSimAlgos_CaloVHitCorrection_h
#define CaloSimAlgos_CaloVHitCorrection_h

#include "SimDataFormats/CaloHit/interface/PCaloHit.h"

namespace CLHEP {
  class HepRandomEngine;
}

class CaloVHitCorrection {
public:
  virtual ~CaloVHitCorrection() = default;
  virtual double delay(const PCaloHit & hit, CLHEP::HepRandomEngine*) const = 0;
};

#endif

