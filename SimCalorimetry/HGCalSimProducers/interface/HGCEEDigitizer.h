#ifndef SimCalorimetry_HGCSimProducers_hgceedigitizer
#define SimCalorimetry_HGCSimProducers_hgceedigitizer

#include "SimCalorimetry/HGCalSimProducers/interface/HGCDigitizerBase.h"
#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"

class HGCEEDigitizer : public HGCDigitizerBase {
public:
  HGCEEDigitizer(const edm::ParameterSet& ps);
  void runDigitizer(std::unique_ptr<HGCalDigiCollection>& digiColl,
                    hgc::HGCSimHitDataAccumulator& simData,
                    const CaloSubdetectorGeometry* theGeom,
                    const std::unordered_set<DetId>& validIds,
                    CLHEP::HepRandomEngine* engine) override;
  ~HGCEEDigitizer() override;

private:
};

#endif
