#ifndef SimCalorimetry_HGCSimProducers_hgchefrontdigitizer
#define SimCalorimetry_HGCSimProducers_hgchefrontdigitizer

#include "SimCalorimetry/HGCSimProducers/interface/HGCDigitizerBase.h"
#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"

class HGCHEfrontDigitizer : public HGCDigitizerBase<HGCHEDataFrame>
{
 public:
  HGCHEfrontDigitizer(const edm::ParameterSet& ps);
  void runDigitizer(std::auto_ptr<HGCHEDigiCollection> &digiColl,HGCSimHitDataAccumulator &simData);
  ~HGCHEfrontDigitizer();
 private:

};

#endif 
