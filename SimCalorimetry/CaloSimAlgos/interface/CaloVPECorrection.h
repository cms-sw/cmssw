#ifndef CaloSimAlgos_CaloVPECorrection_h
#define CaloSimAlgos_CaloVPECorrection_h

/* Corrections to the number of photoelectrons
   applied after the Poisson statistics.  Handles
   effects like quantum efficiency or ion feedback
*/
#include "DataFormats/DetId/interface/DetId.h"

class CaloVPECorrection
{
public:
  virtual double correctPE(const DetId & detId, double npe) const = 0;
};

#endif

