// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "RecoTracker/PixelTrackFitting/interface/PixelTrackCleanerBySharedHits.h"

class PixelTrackCleanerBySharedHitsESProducer : public edm::ESProducer {
public:
  PixelTrackCleanerBySharedHitsESProducer(const edm::ParameterSet& iConfig);
  ~PixelTrackCleanerBySharedHitsESProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  std::unique_ptr<PixelTrackCleaner> produce(const PixelTrackCleaner::Record& iRecord);

private:
  const bool useQuadrupletAlgo_;
};

PixelTrackCleanerBySharedHitsESProducer::PixelTrackCleanerBySharedHitsESProducer(const edm::ParameterSet& iConfig)
    : useQuadrupletAlgo_(iConfig.getParameter<bool>("useQuadrupletAlgo")) {
  auto componentName = iConfig.getParameter<std::string>("ComponentName");
  setWhatProduced(this, componentName);
}

void PixelTrackCleanerBySharedHitsESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("ComponentName", "pixelTrackCleanerBySharedHits");
  desc.add<bool>("useQuadrupletAlgo", false);
  descriptions.add("pixelTrackCleanerBySharedHits", desc);
}

std::unique_ptr<PixelTrackCleaner> PixelTrackCleanerBySharedHitsESProducer::produce(
    const PixelTrackCleaner::Record& iRecord) {
  return std::make_unique<PixelTrackCleanerBySharedHits>(useQuadrupletAlgo_);
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_EVENTSETUP_MODULE(PixelTrackCleanerBySharedHitsESProducer);
