#ifndef SimCalorimetry_HGCSimProducers_hgchebackdigitizer
#define SimCalorimetry_HGCSimProducers_hgchebackdigitizer

#include "SimCalorimetry/HGCalSimProducers/interface/HGCDigitizerBase.h"
#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"

class HGCHEbackDigitizer : public HGCDigitizerBase<HGCBHDataFrame>
{
 public:

  HGCHEbackDigitizer(const edm::ParameterSet& ps);
  void runDigitizer(std::unique_ptr<HGCBHDigiCollection> &digiColl,hgc::HGCSimHitDataAccumulator &simData,
		    const CaloSubdetectorGeometry* theGeom, const std::unordered_set<DetId>& validIds,
		    uint32_t digitizationType, CLHEP::HepRandomEngine* engine) override;
  ~HGCHEbackDigitizer() override;

 private:

  //calice-like digitization parameters
  float keV2MIP_, noise_MIP_;
  float nPEperMIP_, nTotalPE_, xTalk_, sdPixels_;
  void runCaliceLikeDigitizer(std::unique_ptr<HGCBHDigiCollection> &digiColl,hgc::HGCSimHitDataAccumulator &simData, 
			      const CaloSubdetectorGeometry* theGeom, const std::unordered_set<DetId>& validIds,
			      CLHEP::HepRandomEngine* engine);
};

#endif 
