#ifndef TrackingTools_ESProducers_TkTransientTrackingRecHitBuilderESProducer_h
#define TrackingTools_ESProducers_TkTransientTrackingRecHitBuilderESProducer_h

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "RecoLocalTracker/Records/interface/TrackerCPERecord.h"

#include <memory>

class TkTransientTrackingRecHitBuilderESProducer : public edm::ESProducer {
public:
  TkTransientTrackingRecHitBuilderESProducer(const edm::ParameterSet &p);
  ~TkTransientTrackingRecHitBuilderESProducer() override;
  std::unique_ptr<TransientTrackingRecHitBuilder> produce(const TransientRecHitRecord &);

private:
  edm::ParameterSet pset_;
};

#endif
