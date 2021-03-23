#include "SimCalorimetry/HGCalSimProducers/interface/HGCDigitizerBase.h"
#include "SimCalorimetry/HGCalSimProducers/interface/HGCDigitizerPluginFactory.h"
#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"

using namespace hgc_digi;

class HFNoseDigitizer : public HGCDigitizerBase {
public:
  HFNoseDigitizer(const edm::ParameterSet& ps);
  void runDigitizer(std::unique_ptr<HGCalDigiCollection>& digiColl,
                    hgc::HGCSimHitDataAccumulator& simData,
                    const CaloSubdetectorGeometry* theGeom,
                    const std::unordered_set<DetId>& validIds,
                    CLHEP::HepRandomEngine* engine) override;
  ~HFNoseDigitizer() override;

private:
};

HFNoseDigitizer::HFNoseDigitizer(const edm::ParameterSet& ps) : HGCDigitizerBase(ps) {
  this->det_ = DetId::Forward;
  this->subdet_ = ForwardSubdetector::HFNose;
}

//
void HFNoseDigitizer::runDigitizer(std::unique_ptr<HGCalDigiCollection>& digiColl,
                                   HGCSimHitDataAccumulator& simData,
                                   const CaloSubdetectorGeometry* theGeom,
                                   const std::unordered_set<DetId>& validIds,
                                   CLHEP::HepRandomEngine* engine) {}

//
HFNoseDigitizer::~HFNoseDigitizer() {}

DEFINE_EDM_PLUGIN(HGCDigitizerPluginFactory, HFNoseDigitizer, "HFNoseDigitizer");
