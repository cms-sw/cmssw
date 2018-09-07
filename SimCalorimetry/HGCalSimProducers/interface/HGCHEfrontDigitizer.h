#ifndef SimCalorimetry_HGCSimProducers_hgchefrontdigitizer
#define SimCalorimetry_HGCSimProducers_hgchefrontdigitizer

#include "SimCalorimetry/HGCalSimProducers/interface/HGCDigitizerBase.h"
#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"

class HGCHEfrontDigitizer : public HGCDigitizerBase<HGCalDataFrame> {

public:
  HGCHEfrontDigitizer(const edm::ParameterSet& ps);
  void runDigitizer(std::unique_ptr<HGCalDigiCollection> &digiColl, hgc::HGCSimHitDataAccumulator &simData,
		    const CaloSubdetectorGeometry* theGeom, const std::unordered_set<DetId>& validIds,
		    uint32_t digitizationType, CLHEP::HepRandomEngine* engine) override;
  ~HGCHEfrontDigitizer() override;
private:

};

#endif 
