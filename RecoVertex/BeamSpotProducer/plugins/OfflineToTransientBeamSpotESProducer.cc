#include "RecoVertex/BeamSpotProducer/plugins/OfflineToTransientBeamSpotESProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Utilities/interface/do_nothing_deleter.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include <iostream>
#include <memory>
#include <string>

using namespace edm;

OfflineToTransientBeamSpotESProducer::OfflineToTransientBeamSpotESProducer(const edm::ParameterSet& p) {
  auto cc = setWhatProduced(this);

  bsOfflineToken_ = cc.consumesFrom<BeamSpotObjects, BeamSpotObjectsRcd>();
}

OfflineToTransientBeamSpotESProducer::~OfflineToTransientBeamSpotESProducer() {
  //delete theOfflineBS_;
}
void OfflineToTransientBeamSpotESProducer::fillDescription(edm::ConfigurationDescriptions& desc) {
  edm::ParameterSetDescription dsc;
  desc.addWithDefaultLabel(dsc);
}
std::shared_ptr<const BeamSpotObjects> OfflineToTransientBeamSpotESProducer::produce(
    const BeamSpotTransientObjectsRcd& iRecord) {
  auto optionalRec = iRecord.tryToGetRecord<BeamSpotObjectsRcd>();
  if (not optionalRec) {
    return std::shared_ptr<const BeamSpotObjects>(&dummyBS_, edm::do_nothing_deleter());
  }
  return std::shared_ptr<const BeamSpotObjects>(&optionalRec->get(bsOfflineToken_), edm::do_nothing_deleter());
};

DEFINE_FWK_EVENTSETUP_MODULE(OfflineToTransientBeamSpotESProducer);
