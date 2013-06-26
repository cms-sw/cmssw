#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "SimMuon/RPCDigitizer/src/RPCSimSetUp.h"
#include "SimMuon/RPCDigitizer/src/RPCDigiProducer.h"
#include "SimMuon/RPCDigitizer/src/RPCDigitizer.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimMuon/RPCDigitizer/src/RPCSynchronizer.h"
#include <sstream>
#include <string>

#include <map>
#include <vector>

#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"

//Random Number
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "CLHEP/Random/RandomEngine.h"



RPCDigiProducer::RPCDigiProducer(const edm::ParameterSet& ps) {

  produces<RPCDigiCollection>();
  produces<RPCDigitizerSimLinks>("RPCDigiSimLink");

  //Name of Collection used for create the XF 
  mix_ = ps.getParameter<std::string>("mixLabel");
  collection_for_XF = ps.getParameter<std::string>("InputCollection");

  edm::Service<edm::RandomNumberGenerator> rng;
  if ( ! rng.isAvailable()) {
   throw cms::Exception("Configuration")
     << "RPCDigitizer requires the RandomNumberGeneratorService\n"
        "which is not present in the configuration file.  You must add the service\n"
        "in the configuration file or remove the modules that require it.";
  }


  CLHEP::HepRandomEngine& engine = rng->getEngine();

  theRPCSimSetUp =  new RPCSimSetUp(ps);
  theDigitizer = new RPCDigitizer(ps,engine);


}

RPCDigiProducer::~RPCDigiProducer() {
  delete theDigitizer;
  delete theRPCSimSetUp;
}

void RPCDigiProducer::beginRun(const edm::Run& r, const edm::EventSetup& eventSetup){

  edm::ESHandle<RPCGeometry> hGeom;
  eventSetup.get<MuonGeometryRecord>().get( hGeom );
  const RPCGeometry *pGeom = &*hGeom;

  edm::ESHandle<RPCStripNoises> noiseRcd;
  eventSetup.get<RPCStripNoisesRcd>().get(noiseRcd);

   edm::ESHandle<RPCClusterSize> clsRcd;
   eventSetup.get<RPCClusterSizeRcd>().get(clsRcd);

   theRPCSimSetUp->setRPCSetUp(noiseRcd->getVNoise(), clsRcd->getCls());
//    theRPCSimSetUp->setRPCSetUp(noiseRcd->getVNoise(), noiseRcd->getCls());
  
  theDigitizer->setGeometry( pGeom );
  theRPCSimSetUp->setGeometry( pGeom );
  theDigitizer->setRPCSimSetUp( theRPCSimSetUp );
}

void RPCDigiProducer::produce(edm::Event& e, const edm::EventSetup& eventSetup) {

  edm::Handle<CrossingFrame<PSimHit> > cf;
  e.getByLabel(mix_, collection_for_XF, cf);

  std::auto_ptr<MixCollection<PSimHit> > 
    hits( new MixCollection<PSimHit>(cf.product()) );

  // Create empty output
  std::auto_ptr<RPCDigiCollection> pDigis(new RPCDigiCollection());
  std::auto_ptr<RPCDigitizerSimLinks> RPCDigitSimLink(new RPCDigitizerSimLinks() );

  // run the digitizer
  theDigitizer->doAction(*hits, *pDigis, *RPCDigitSimLink);

  // store them in the event
  e.put(pDigis);
  e.put(RPCDigitSimLink,"RPCDigiSimLink");
}

