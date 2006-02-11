#ifndef CaloSimAlgos_CaloVHitCorrection_h
#define CaloSimAlgos_CaloVHitCorrection_h

#include "SimDataFormats/CaloHit/interface/PCaloHit.h"

class CaloVHitCorrection {
public:
  virtual void correct(PCaloHit & hit) const = 0;
};

#endif

