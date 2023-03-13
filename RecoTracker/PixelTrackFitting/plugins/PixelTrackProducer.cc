#include <vector>

#include "DataFormats/Common/interface/OrphanHandle.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/TrajectoryState/interface/LocalTrajectoryParameters.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "RecoTracker/PixelTrackFitting/interface/PixelTrackReconstruction.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "storeTracks.h"

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

namespace edm {
  class Event;
  class EventSetup;
  class ParameterSet;
  class ConfigurationDescriptions;
}  // namespace edm
class TrackerTopology;

using namespace pixeltrackfitting;
using edm::ParameterSet;

class PixelTrackProducer : public edm::stream::EDProducer<> {
public:
  explicit PixelTrackProducer(const edm::ParameterSet& cfg)
      : theReconstruction(cfg, consumesCollector()), htTopoToken_(esConsumes()) {
    edm::LogInfo("PixelTrackProducer") << " construction...";
    produces<TrackingRecHitCollection>();
    produces<reco::TrackExtraCollection>();
    // TrackCollection refers to TrackingRechit and TrackExtra
    // collections, need to declare its production after them to work
    // around a rare race condition in framework scheduling
    produces<reco::TrackCollection>();
  }

  ~PixelTrackProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;

    desc.add<std::string>("passLabel", "pixelTracks");  // What is this? It is not used anywhere in this code.
    PixelTrackReconstruction::fillDescriptions(desc);

    descriptions.add("pixelTracks", desc);
  }

  void produce(edm::Event& ev, const edm::EventSetup& es) override {
    LogDebug("PixelTrackProducer, produce") << "event# :" << ev.id();

    TracksWithTTRHs tracks;
    theReconstruction.run(tracks, ev, es);
    auto htTopo = es.getData(htTopoToken_);

    // store tracks
    storeTracks(ev, tracks, htTopo);
  }

private:
  PixelTrackReconstruction theReconstruction;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> htTopoToken_;
};

DEFINE_FWK_MODULE(PixelTrackProducer);
