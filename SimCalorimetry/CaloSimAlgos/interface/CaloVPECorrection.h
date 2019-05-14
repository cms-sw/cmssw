#ifndef CaloSimAlgos_CaloVPECorrection_h
#define CaloSimAlgos_CaloVPECorrection_h

/* Corrections to the number of photoelectrons
   applied after the Poisson statistics.  Handles
   effects like quantum efficiency or ion feedback
*/
#include "DataFormats/DetId/interface/DetId.h"

namespace CLHEP {
  class HepRandomEngine;
}

class CaloVPECorrection {
public:
  virtual ~CaloVPECorrection() {}
  virtual double correctPE(const DetId &detId, double npe, CLHEP::HepRandomEngine *) const = 0;
};

#endif
