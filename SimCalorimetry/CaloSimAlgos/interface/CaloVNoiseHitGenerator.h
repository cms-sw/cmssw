#ifndef CaloSimAlgos_CaloVNoiseHitGenerator_h
#define CaloSimAlgos_CaloVNoiseHitGenerator_h

#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include<vector>

class CaloVNoiseHitGenerator
{
public:
  virtual void getNoiseHits(std::vector<PCaloHit> & noiseHits) = 0;
};

#endif

