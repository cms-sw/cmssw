#include "SimMuon/GEMDigitizer/interface/GEMDigiModule.h"

#include "SimMuon/GEMDigitizer/interface/GEMSignalModel.h"
#include "SimMuon/GEMDigitizer/interface/GEMBkgModel.h"
#include "SimMuon/GEMDigitizer/interface/GEMNoiseModel.h"
#include "Geometry/GEMGeometry/interface/GEMEtaPartition.h"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"

GEMDigiModule::GEMDigiModule(const edm::ParameterSet& config) {
  bool simulateBkgNoise_(config.getParameter<bool>("simulateBkgNoise"));
  bool simulateIntrinsicNoise_(config.getParameter<bool>("simulateIntrinsicNoise"));
  if (simulateIntrinsicNoise_) {
    models.push_back(std::make_unique<GEMNoiseModel>(config));
  }
  if (simulateBkgNoise_) {
    models.push_back(std::make_unique<GEMBkgModel>(config));
  }
  models.push_back(std::make_unique<GEMSignalModel>(config));
}

GEMDigiModule::~GEMDigiModule() = default;

void GEMDigiModule::simulate(const GEMEtaPartition* roll,
                             const edm::PSimHitContainer& simHits,
                             CLHEP::HepRandomEngine* engine) {
  stripDigiSimLinks_.clear();
  detectorHitMap_.clear();
  stripDigiSimLinks_ = StripDigiSimLinks(roll->id().rawId());
  theGemDigiSimLinks_.clear();
  theGemDigiSimLinks_ = GEMDigiSimLinks(roll->id().rawId());
  for (auto&& model : models) {
    model->simulate(roll, simHits, engine, strips_, detectorHitMap_);
  }
  return;
}

void GEMDigiModule::fillDigis(int rollDetId, GEMDigiCollection& digis) {
  for (const auto& d : strips_) {
    if (d.second == -999)
      continue;
    // (strip, bx)
    GEMDigi digi(d.first, d.second);
    digis.insertDigi(GEMDetId(rollDetId), digi);
    addLinks(d.first, d.second);
    addLinksWithPartId(d.first, d.second);
  }
  strips_.clear();
}

void GEMDigiModule::addLinks(unsigned int strip, int bx) {
  std::pair<unsigned int, int> digi(strip, bx);
  auto channelHitItr = detectorHitMap_.equal_range(digi);

  // find the fraction contribution for each SimTrack
  std::map<int, float> simTrackChargeMap;
  std::map<int, EncodedEventId> eventIdMap;
  float totalCharge(0.);
  for (auto hitItr = channelHitItr.first; hitItr != channelHitItr.second; ++hitItr) {
    const PSimHit* hit(hitItr->second);
    // might be zero for unit tests and such
    if (hit == nullptr)
      continue;

    int simTrackId(hit->trackId());
    //float charge = hit->getCharge();
    const float charge(1.f);
    auto chargeItr = simTrackChargeMap.find(simTrackId);
    if (chargeItr == simTrackChargeMap.end()) {
      simTrackChargeMap[simTrackId] = charge;
      eventIdMap[simTrackId] = hit->eventId();
    } else {
      chargeItr->second += charge;
    }
    totalCharge += charge;
  }

  for (const auto& charge : simTrackChargeMap) {
    const int simTrackId(charge.first);
    auto link(StripDigiSimLink(strip, simTrackId, eventIdMap[simTrackId], charge.second / totalCharge));
    stripDigiSimLinks_.push_back(link);
  }
}

void GEMDigiModule::addLinksWithPartId(unsigned int strip, int bx) {
  std::pair<unsigned int, int> digi(strip, bx);
  std::pair<DetectorHitMap::iterator, DetectorHitMap::iterator> channelHitItr = detectorHitMap_.equal_range(digi);

  for (DetectorHitMap::iterator hitItr = channelHitItr.first; hitItr != channelHitItr.second; ++hitItr) {
    const PSimHit* hit = (hitItr->second);
    // might be zero for unit tests and such
    if (hit == nullptr)
      continue;

    theGemDigiSimLinks_.push_back(GEMDigiSimLink(digi,
                                                 hit->entryPoint(),
                                                 hit->momentumAtEntry(),
                                                 hit->timeOfFlight(),
                                                 hit->energyLoss(),
                                                 hit->particleType(),
                                                 hit->detUnitId(),
                                                 hit->trackId(),
                                                 hit->eventId(),
                                                 hit->processType()));
  }
}

void GEMDigiModule::setGeometry(const GEMGeometry* geom) {
  for (auto&& model : models) {
    model->setGeometry(geom);
  }
}
