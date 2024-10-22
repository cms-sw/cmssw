#ifndef RecoTracker_PixelLowPtUtilities_plugins_PixelVertexProducerClusters_h
#define RecoTracker_PixelLowPtUtilities_plugins_PixelVertexProducerClusters_h

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"

namespace edm {
  class Run;
  class Event;
  class EventSetup;
}  // namespace edm

class TrackerGeometry;

class PixelVertexProducerClusters : public edm::global::EDProducer<> {
public:
  explicit PixelVertexProducerClusters(const edm::ParameterSet& ps);
  ~PixelVertexProducerClusters() override;
  void produce(edm::StreamID, edm::Event& ev, const edm::EventSetup& es) const override;

private:
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  edm::EDGetTokenT<SiPixelRecHitCollection> pixelToken_;
};
#endif
