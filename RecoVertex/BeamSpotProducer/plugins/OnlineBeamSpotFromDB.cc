/**_________________________________________________________________
   class:   OnlineBeamSpotFromDB.cc
   package: RecoVertex/BeamSpotProducer
   


 author: Francisco Yumiceva, Fermilab (yumiceva@fnal.gov)


________________________________________________________________**/

// C++ standard
#include <string>
// CMS
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "RecoVertex/BeamSpotProducer/interface/OnlineBeamSpotFromDB.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/IOVSyncValue.h"
#include "CondFormats/DataRecord/interface/BeamSpotTransientObjectsRcd.h"
#include "CondFormats/BeamSpotObjects/interface/BeamSpotOnlineObjects.h"
#include "CondFormats/BeamSpotObjects/interface/BeamSpotObjects.h"

OnlineBeamSpotFromDB::OnlineBeamSpotFromDB(const edm::ParameterSet& iConfig) {}

OnlineBeamSpotFromDB::~OnlineBeamSpotFromDB() {}

void OnlineBeamSpotFromDB::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::ESHandle<BeamSpotObjects> beamGThandle;
  edm::ESHandle<BeamSpotObjects> beamhandle;
  iSetup.get<BeamSpotTransientObjectsRcd>().get(beamhandle);
  const BeamSpotObjects* mybeamspot = beamhandle.product();
  iSetup.get<BeamSpotObjectsRcd>().get(beamGThandle);
  const BeamSpotObjects* myGTbeamspot = beamGThandle.product();


  std::cout << " for runs: " << iEvent.id().run() << " - " << iEvent.id().run() << std::endl;
  //std::cout << iEvent.getRun().beginTime().value() << std::endl;
  //std::cout << iEvent.time().value() << std::endl;
  std::cout << *mybeamspot << std::endl;
  std::cout<<*myGTbeamspot<<std::endl;
}

void OnlineBeamSpotFromDB::beginJob() {}

void OnlineBeamSpotFromDB::endJob() {}

//define this as a plug-in
DEFINE_FWK_MODULE(OnlineBeamSpotFromDB);
