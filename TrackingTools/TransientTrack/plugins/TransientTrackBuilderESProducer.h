#ifndef TrackingTools_ESProducers_TransientTrackBuilderESProducer_h
#define TrackingTools_ESProducers_TransientTrackBuilderESProducer_h

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"

#include <memory>

class TransientTrackBuilderESProducer : public edm::ESProducer {
public:
  TransientTrackBuilderESProducer(const edm::ParameterSet &p);

  std::unique_ptr<TransientTrackBuilder> produce(const TransientTrackRecord &);

private:
  TransientTrackBuilderESProducer(edm::ESConsumesCollector &&c);

  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magToken_;
  const edm::ESGetToken<GlobalTrackingGeometry, GlobalTrackingGeometryRecord> geomToken_;
};

#endif
