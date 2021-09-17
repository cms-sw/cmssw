#include "TransientTrackBuilderESProducer.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include <string>
#include <memory>

using namespace edm;

TransientTrackBuilderESProducer::TransientTrackBuilderESProducer(const edm::ParameterSet& p) {
  auto cc = setWhatProduced(this, p.getParameter<std::string>("ComponentName"));
  magToken_ = cc.consumes();
  geomToken_ = cc.consumes();
}

std::unique_ptr<TransientTrackBuilder> TransientTrackBuilderESProducer::produce(const TransientTrackRecord& iRecord) {
  return std::make_unique<TransientTrackBuilder>(&iRecord.get(magToken_), iRecord.getHandle(geomToken_));
}

void TransientTrackBuilderESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("ComponentName", "TransientTrackBuilder")
      ->setComment("data label to use when getting the data product");

  descriptions.addDefault(desc);
}
