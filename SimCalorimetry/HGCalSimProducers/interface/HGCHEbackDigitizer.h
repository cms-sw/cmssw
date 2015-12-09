#ifndef SimCalorimetry_HGCSimProducers_hgchebackdigitizer
#define SimCalorimetry_HGCSimProducers_hgchebackdigitizer

#include "SimCalorimetry/HGCalSimProducers/interface/HGCDigitizerBase.h"
#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"

class HGCHEbackDigitizer : public HGCDigitizerBase<HGCHEDataFrame>
{
 public:

  HGCHEbackDigitizer(const edm::ParameterSet& ps);
  void runDigitizer(std::auto_ptr<HGCHEDigiCollection> &digiColl,hgc::HGCSimHitDataAccumulator &simData,uint32_t digitizationType, CLHEP::HepRandomEngine* engine);
  ~HGCHEbackDigitizer();

 private:

  //calice-like digitization parameters
  float keV2MIP_, noise_MIP_;
  float nPEperMIP_, nTotalPE_, xTalk_, sdPixels_;
  void runCaliceLikeDigitizer(std::auto_ptr<HGCHEDigiCollection> &digiColl,hgc::HGCSimHitDataAccumulator &simData, CLHEP::HepRandomEngine* engine);
};

#endif 
