#include "RecoVertex/BeamSpotProducer/plugins/OnlineBeamSpotESProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Utilities/interface/do_nothing_deleter.h"

#include <iostream>
#include <memory>
#include <string>

using namespace edm;

OnlineBeamSpotESProducer::OnlineBeamSpotESProducer(const edm::ParameterSet& p){

//label_HLT_ = p.getParameter<std::string>("HLTLabel");

setWhatProduced(this);
newHLT_ = false;
newLegacy_ = false;


theHLTBS_ = new BeamSpotOnlineObjects;
theLegacyBS_ = new BeamSpotOnlineObjects;

bsHLTToken_ = setWhatProduced(this).consumesFrom<BeamSpotOnlineObjects, BeamSpotOnlineHLTObjectsRcd>();
bsLegacyToken_ = setWhatProduced(this).consumesFrom<BeamSpotOnlineObjects, BeamSpotOnlineLegacyObjectsRcd>();


          
}

OnlineBeamSpotESProducer::~OnlineBeamSpotESProducer() {
delete theHLTBS_;
delete theLegacyBS_;
}


std::shared_ptr<const BeamSpotOnlineObjects> OnlineBeamSpotESProducer::produce(const BeamSpotTransientObjectsRcd& iRecord) {
  auto host = holder_.makeOrGet([]() {
        return new HostType;
  });

  host->ifRecordChanges<BeamSpotOnlineHLTObjectsRcd>(iRecord,
                                           [this, h=host.get()](auto const& rec) {
      newHLT_ = true;                                       
      theHLTBS_ = &rec.get(bsHLTToken_);
      });
  host->ifRecordChanges<BeamSpotOnlineLegacyObjectsRcd>(iRecord,
                                           [this, h=host.get()](auto const& rec) {
      newLegacy_ = true;
      theLegacyBS_ = &rec.get(bsLegacyToken_);
      });

   if (newHLT_ || newLegacy_){
    //compare the HLT with Legacy BS values to choose what to use:

   };

    
  return std::shared_ptr<const BeamSpotOnlineObjects>(&(*theHLTBS_), edm::do_nothing_deleter());

 };

DEFINE_FWK_EVENTSETUP_MODULE(OnlineBeamSpotESProducer);