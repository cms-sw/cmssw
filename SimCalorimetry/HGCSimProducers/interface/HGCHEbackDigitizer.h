#ifndef SimCalorimetry_HGCSimProducers_hgchebackdigitizer
#define SimCalorimetry_HGCSimProducers_hgchebackdigitizer

#include "SimCalorimetry/HGCSimProducers/interface/HGCDigitizerBase.h"
#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"

class HGCHEbackDigitizer : public HGCDigitizerBase<HGCHEDataFrame>
{
 public:
  HGCHEbackDigitizer(const edm::ParameterSet& ps);
  void runDigitizer(std::auto_ptr<HGCHEDigiCollection> &digiColl,HGCSimHitDataAccumulator &simData);
  ~HGCHEbackDigitizer();
 private:

};

#endif 
