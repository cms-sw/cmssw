#include "TransientTrackBuilderESProducer.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "boost/mpl/vector.hpp"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include <string>
#include <memory>

using namespace edm;

TransientTrackBuilderESProducer::TransientTrackBuilderESProducer(const edm::ParameterSet& p)
    : TransientTrackBuilderESProducer(setWhatProduced(this, p.getParameter<std::string>("ComponentName"))) {}

TransientTrackBuilderESProducer::TransientTrackBuilderESProducer(edm::ESConsumesCollector&& c)
    : magToken_(c.consumesFrom<MagneticField, IdealMagneticFieldRecord>()),
      geomToken_(c.consumesFrom<GlobalTrackingGeometry, GlobalTrackingGeometryRecord>()) {}

std::unique_ptr<TransientTrackBuilder> TransientTrackBuilderESProducer::produce(const TransientTrackRecord& iRecord) {
  return std::make_unique<TransientTrackBuilder>(&iRecord.get(magToken_), iRecord.getHandle(geomToken_));
}
