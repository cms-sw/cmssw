#ifndef CaloVHitFilter_h
#define CaloVHitFilter_h

#include "SimDataFormats/CaloHit/interface/PCaloHit.h"


class CaloVHitFilter {
public:
  virtual bool accepts(const PCaloHit & hit) const = 0;
};

#endif

