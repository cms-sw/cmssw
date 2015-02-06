#ifndef SimCalorimetry_HGCSimProducers_hgchebackdigitizer
#define SimCalorimetry_HGCSimProducers_hgchebackdigitizer

#include "SimCalorimetry/HGCSimProducers/interface/HGCDigitizerBase.h"
#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"

#include "CLHEP/Random/RandPoisson.h"

class HGCHEbackDigitizer : public HGCDigitizerBase<HGCHEDataFrame>
{
 public:

  HGCHEbackDigitizer(const edm::ParameterSet& ps);
  void setRandomNumberEngine(CLHEP::HepRandomEngine& engine);
  void runDigitizer(std::auto_ptr<HGCHEDigiCollection> &digiColl,HGCSimHitDataAccumulator &simData,uint32_t digitizationType);
  ~HGCHEbackDigitizer();

 private:

  //calice-like digitization parameters
  float nPEperMIP_, nTotalPE_, xTalk_, sdPixels_, lsbInMIP_;
  mutable CLHEP::RandPoisson *peGen_;
  mutable CLHEP::RandGauss *sigGen_;
  void runCaliceLikeDigitizer(std::auto_ptr<HGCHEDigiCollection> &digiColl,HGCSimHitDataAccumulator &simData);
};

#endif 
