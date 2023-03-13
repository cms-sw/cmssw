// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include "RecoTracker/PixelLowPtUtilities/interface/TrackCleaner.h"

class TrackCleanerESProducer : public edm::ESProducer {
public:
  TrackCleanerESProducer(const edm::ParameterSet& iConfig);
  ~TrackCleanerESProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  std::unique_ptr<PixelTrackCleaner> produce(const PixelTrackCleaner::Record& iRecord);

private:
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> trackerToken_;
};

TrackCleanerESProducer::TrackCleanerESProducer(const edm::ParameterSet& iConfig)
    : trackerToken_(setWhatProduced(this, iConfig.getParameter<std::string>("ComponentName"))
                        .consumesFrom<TrackerTopology, TrackerTopologyRcd>()) {}

void TrackCleanerESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("ComponentName", "trackCleaner");
  descriptions.add("trackCleaner", desc);
}

std::unique_ptr<PixelTrackCleaner> TrackCleanerESProducer::produce(const PixelTrackCleaner::Record& iRecord) {
  return std::make_unique<TrackCleaner>(&iRecord.get(trackerToken_));
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_EVENTSETUP_MODULE(TrackCleanerESProducer);
