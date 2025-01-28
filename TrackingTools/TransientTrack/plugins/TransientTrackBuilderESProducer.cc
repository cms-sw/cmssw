// system includes
#include <memory>
#include <string>

// user includes
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"

class TransientTrackBuilderESProducer : public edm::ESProducer {
public:
  TransientTrackBuilderESProducer(const edm::ParameterSet& p);

  std::unique_ptr<TransientTrackBuilder> produce(const TransientTrackRecord&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magToken_;
  edm::ESGetToken<GlobalTrackingGeometry, GlobalTrackingGeometryRecord> geomToken_;
};

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

  descriptions.addWithDefaultLabel(desc);
}

#include "FWCore/Framework/interface/ModuleFactory.h"
DEFINE_FWK_EVENTSETUP_MODULE(TransientTrackBuilderESProducer);
