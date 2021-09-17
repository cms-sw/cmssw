#include "SimCalorimetry/HGCalSimProducers/interface/HGCDigitizerBase.h"
#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "SimCalorimetry/HGCalSimProducers/interface/HGCDigitizerPluginFactory.h"

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

using namespace hgc_digi;

//
HGCEEDigitizer::HGCEEDigitizer(const edm::ParameterSet& ps) : HGCDigitizerBase(ps) { this->det_ = DetId::HGCalEE; }

//
void HGCEEDigitizer::runDigitizer(std::unique_ptr<HGCalDigiCollection>& digiColl,
                                  HGCSimHitDataAccumulator& simData,
                                  const CaloSubdetectorGeometry* theGeom,
                                  const std::unordered_set<DetId>& validIds,
                                  CLHEP::HepRandomEngine* engine) {}

//
HGCEEDigitizer::~HGCEEDigitizer() {}

DEFINE_EDM_PLUGIN(HGCDigitizerPluginFactory, HGCEEDigitizer, "HGCEEDigitizer");
