#include "RecoVertex/BeamSpotProducer/plugins/OfflineToTransientBeamSpotESProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Utilities/interface/do_nothing_deleter.h"

#include <iostream>
#include <memory>
#include <string>

using namespace edm;

OfflineToTransientBeamSpotESProducer::OfflineToTransientBeamSpotESProducer(const edm::ParameterSet& p) {
  auto cc = setWhatProduced(this);

  transientBS_ = new BeamSpotObjects;
  //theOfflineBS_ = new BeamSpotObjects;
  bsOfflineToken_ = cc.consumesFrom<BeamSpotObjects, BeamSpotObjectsRcd>();
}

OfflineToTransientBeamSpotESProducer::~OfflineToTransientBeamSpotESProducer() {
  delete transientBS_;
  //delete theOfflineBS_;
}

std::shared_ptr<const BeamSpotObjects> OfflineToTransientBeamSpotESProducer::produce(
    const BeamSpotTransientObjectsRcd& iRecord) {
  if (!(iRecord.tryToGetRecord<BeamSpotObjectsRcd>())) {
    //Missing offline record????

    return std::shared_ptr<const BeamSpotObjects>(&(*transientBS_), edm::do_nothing_deleter());
  }

  auto host = holder_.makeOrGet([]() { return new HostType; });

  if (iRecord.tryToGetRecord<BeamSpotObjectsRcd>()) {
    host->ifRecordChanges<BeamSpotObjectsRcd>(
        iRecord, [this, h = host.get()](auto const& rec) { transientBS_ = &rec.get(bsOfflineToken_); });
  }

  std::cout << "Transient " << *transientBS_ << std::endl;

  return std::shared_ptr<const BeamSpotObjects>(&(*transientBS_), edm::do_nothing_deleter());
};

DEFINE_FWK_EVENTSETUP_MODULE(OfflineToTransientBeamSpotESProducer);
