#ifndef HcalTestHitGenerator_h
#define HcalTestHitGenerator_h

#include "SimCalorimetry/CaloSimAlgos/interface/CaloVNoiseHitGenerator.h"
#include <vector>
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class HcalTestHitGenerator : public CaloVNoiseHitGenerator
{
public:
  explicit HcalTestHitGenerator(const edm::ParameterSet & ps);
  virtual ~HcalTestHitGenerator() {}
  virtual void getNoiseHits(std::vector<PCaloHit> & noiseHits);
private:
  std::vector<double> theBarrelSampling;
  std::vector<double> theEndcapSampling;
};

#endif

