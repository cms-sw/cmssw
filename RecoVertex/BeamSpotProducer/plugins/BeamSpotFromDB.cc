/**_________________________________________________________________
   class:   BeamSpotFromDB.cc
   package: RecoVertex/BeamSpotProducer
   


 author: Francisco Yumiceva, Fermilab (yumiceva@fnal.gov)


________________________________________________________________**/

// C++ standard
#include <string>
// CMS
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "RecoVertex/BeamSpotProducer/interface/BeamSpotFromDB.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/IOVSyncValue.h"

BeamSpotFromDB::BeamSpotFromDB(const edm::ParameterSet& iConfig)
    : m_beamToken(esConsumes<BeamSpotObjects, BeamSpotObjectsRcd>()) {}

BeamSpotFromDB::~BeamSpotFromDB() {}

void BeamSpotFromDB::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::ESHandle<BeamSpotObjects> beamhandle = iSetup.getHandle(m_beamToken);
  const BeamSpotObjects* mybeamspot = beamhandle.product();

  std::cout << " for runs: " << iEvent.id().run() << " - " << iEvent.id().run() << std::endl;
  //std::cout << iEvent.getRun().beginTime().value() << std::endl;
  //std::cout << iEvent.time().value() << std::endl;
  std::cout << *mybeamspot << std::endl;
}

void BeamSpotFromDB::beginJob() {}

void BeamSpotFromDB::endJob() {}

//define this as a plug-in
DEFINE_FWK_MODULE(BeamSpotFromDB);
