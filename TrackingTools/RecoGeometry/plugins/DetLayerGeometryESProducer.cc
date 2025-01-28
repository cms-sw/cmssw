#include <memory>
#include <string>

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TrackingTools/DetLayers/interface/DetLayerGeometry.h"
#include "TrackingTools/RecoGeometry/interface/RecoGeometryRecord.h"

class DetLayerGeometryESProducer : public edm::ESProducer {
public:
  DetLayerGeometryESProducer(const edm::ParameterSet& p);
  ~DetLayerGeometryESProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  std::unique_ptr<DetLayerGeometry> produce(const RecoGeometryRecord&);
};

using namespace edm;

DetLayerGeometryESProducer::DetLayerGeometryESProducer(const edm::ParameterSet& p) {
  std::string myName = p.getParameter<std::string>("ComponentName");
  setWhatProduced(this, myName);
}

void DetLayerGeometryESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("ComponentName", std::string(""));
  descriptions.addWithDefaultLabel(desc);
}

std::unique_ptr<DetLayerGeometry> DetLayerGeometryESProducer::produce(const RecoGeometryRecord& iRecord) {
  return std::make_unique<DetLayerGeometry>();
}

DEFINE_FWK_EVENTSETUP_MODULE(DetLayerGeometryESProducer);
