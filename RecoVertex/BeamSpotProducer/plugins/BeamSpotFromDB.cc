/**_________________________________________________________________
   class:   BeamSpotFromDB.cc
   package: RecoVertex/BeamSpotProducer
   


 author: Francisco Yumiceva, Fermilab (yumiceva@fnal.gov)

 version $Id: BeamSpotFromDB.cc,v 1.1 2007/03/30 18:46:57 yumiceva Exp $

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
#include "CondFormats/DataRecord/interface/BeamSpotObjectsRcd.h"
#include "CondFormats/BeamSpotObjects/interface/BeamSpotObjects.h"

BeamSpotFromDB::BeamSpotFromDB(const edm::ParameterSet& iConfig)
{
  
}


BeamSpotFromDB::~BeamSpotFromDB()
{
	
}


void
BeamSpotFromDB::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
	
	edm::ESHandle< BeamSpotObjects > beamhandle;
	iSetup.get<BeamSpotObjectsRcd>().get(beamhandle);
	const BeamSpotObjects *mybeamspot = beamhandle.product();
	
	std::cout << *mybeamspot << std::endl;

}

void
BeamSpotFromDB::beginJob(const edm::EventSetup&)
{
}

void
BeamSpotFromDB::endJob() {
}

//define this as a plug-in
DEFINE_ANOTHER_FWK_MODULE(BeamSpotFromDB);
