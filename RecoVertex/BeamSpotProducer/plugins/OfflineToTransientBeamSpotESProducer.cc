#include "CondFormats/DataRecord/interface/BeamSpotObjectsRcd.h"
#include "CondFormats/BeamSpotObjects/interface/BeamSpotObjects.h"
#include "CondFormats/DataRecord/interface/BeamSpotTransientObjectsRcd.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Framework/interface/ESProductHost.h"
#include "FWCore/Utilities/interface/ReusableObjectHolder.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

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
class OfflineToTransientBeamSpotESProducer : public edm::ESProducer {
public:
  OfflineToTransientBeamSpotESProducer(const edm::ParameterSet& p);
  std::shared_ptr<const BeamSpotObjects> produce(const BeamSpotTransientObjectsRcd&);
  static void fillDescriptions(edm::ConfigurationDescriptions& desc);

private:
  const BeamSpotObjects dummyBS_;
  edm::ESGetToken<BeamSpotObjects, BeamSpotTransientObjectsRcd> const bsToken_;
  edm::ESGetToken<BeamSpotObjects, BeamSpotObjectsRcd> bsOfflineToken_;
};

OfflineToTransientBeamSpotESProducer::OfflineToTransientBeamSpotESProducer(const edm::ParameterSet& p) {
  auto cc = setWhatProduced(this);

  bsOfflineToken_ = cc.consumesFrom<BeamSpotObjects, BeamSpotObjectsRcd>();
}

void OfflineToTransientBeamSpotESProducer::fillDescriptions(edm::ConfigurationDescriptions& desc) {
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
