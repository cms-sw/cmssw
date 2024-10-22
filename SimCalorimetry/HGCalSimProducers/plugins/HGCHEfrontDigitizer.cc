#include "SimCalorimetry/HGCalSimProducers/interface/HGCDigitizerBase.h"
#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"
#include "SimCalorimetry/HGCalSimProducers/interface/HGCDigitizerPluginFactory.h"

using namespace hgc_digi;

class HGCHEfrontDigitizer : public HGCDigitizerBase {
public:
  HGCHEfrontDigitizer(const edm::ParameterSet& ps);
  void runDigitizer(std::unique_ptr<HGCalDigiCollection>& digiColl,
                    hgc::HGCSimHitDataAccumulator& simData,
                    const CaloSubdetectorGeometry* theGeom,
                    const std::unordered_set<DetId>& validIds,
                    CLHEP::HepRandomEngine* engine) override;
  ~HGCHEfrontDigitizer() override;

private:
};

HGCHEfrontDigitizer::HGCHEfrontDigitizer(const edm::ParameterSet& ps) : HGCDigitizerBase(ps) {
  this->det_ = DetId::HGCalHSi;
}

//
void HGCHEfrontDigitizer::runDigitizer(std::unique_ptr<HGCalDigiCollection>& digiColl,
                                       HGCSimHitDataAccumulator& simData,
                                       const CaloSubdetectorGeometry* theGeom,
                                       const std::unordered_set<DetId>& validIds,
                                       CLHEP::HepRandomEngine* engine) {}

//
HGCHEfrontDigitizer::~HGCHEfrontDigitizer() {}
DEFINE_EDM_PLUGIN(HGCDigitizerPluginFactory, HGCHEfrontDigitizer, "HGCHEfrontDigitizer");
