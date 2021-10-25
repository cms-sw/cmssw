#include "FWCore/Framework/interface/EDProducer.h"
#include "SimMuon/RPCDigitizer/src/RPCSimSetUp.h"
#include "SimMuon/RPCDigitizer/src/RPCDigiProducer.h"
#include "SimMuon/RPCDigitizer/src/RPCDigitizer.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
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
#include "CLHEP/Random/RandFlat.h"

namespace CLHEP {
  class HepRandomEngine;
}

RPCDigiProducer::RPCDigiProducer(const edm::ParameterSet& ps) {
  produces<RPCDigiCollection>();
  produces<RPCDigitizerSimLinks>("RPCDigiSimLink");

  //Name of Collection used for create the XF
  mix_ = ps.getParameter<std::string>("mixLabel");
  collection_for_XF = ps.getParameter<std::string>("InputCollection");

  edm::Service<edm::RandomNumberGenerator> rng;
  if (!rng.isAvailable()) {
    throw cms::Exception("Configuration")
        << "RPCDigitizer requires the RandomNumberGeneratorService\n"
           "which is not present in the configuration file.  You must add the service\n"
           "in the configuration file or remove the modules that require it.";
  }
  theRPCSimSetUp = new RPCSimSetUp(ps);
  theDigitizer = new RPCDigitizer(ps);
  crossingFrameToken = consumes<CrossingFrame<PSimHit>>(edm::InputTag(mix_, collection_for_XF));
  geomToken = esConsumes<RPCGeometry, MuonGeometryRecord, edm::Transition::BeginRun>();
  noiseToken = esConsumes<RPCStripNoises, RPCStripNoisesRcd, edm::Transition::BeginRun>();
  clsToken = esConsumes<RPCClusterSize, RPCClusterSizeRcd, edm::Transition::BeginRun>();
}

RPCDigiProducer::~RPCDigiProducer() {
  delete theDigitizer;
  delete theRPCSimSetUp;
}

void RPCDigiProducer::beginRun(const edm::Run& r, const edm::EventSetup& eventSetup) {
  edm::ESHandle<RPCGeometry> hGeom = eventSetup.getHandle(geomToken);
  const RPCGeometry* pGeom = &*hGeom;

  edm::ESHandle<RPCStripNoises> noiseRcd = eventSetup.getHandle(noiseToken);

  edm::ESHandle<RPCClusterSize> clsRcd = eventSetup.getHandle(clsToken);

  theRPCSimSetUp->setGeometry(pGeom);
  theRPCSimSetUp->setRPCSetUp(noiseRcd->getVNoise(), clsRcd->getCls());
  //    theRPCSimSetUp->setRPCSetUp(noiseRcd->getVNoise(), noiseRcd->getCls());

  theDigitizer->setGeometry(pGeom);
  // theRPCSimSetUp->setGeometry( pGeom );
  theDigitizer->setRPCSimSetUp(theRPCSimSetUp);
}

void RPCDigiProducer::produce(edm::Event& e, const edm::EventSetup& eventSetup) {
  edm::Service<edm::RandomNumberGenerator> rng;
  CLHEP::HepRandomEngine* engine = &rng->getEngine(e.streamID());

  LogDebug("RPCDigiProducer") << "[RPCDigiProducer::produce] got the CLHEP::HepRandomEngine engine from the "
                                 "edm::Event.streamID() and edm::Service<edm::RandomNumberGenerator>";
  LogDebug("RPCDigiProducer") << "[RPCDigiProducer::produce] test the CLHEP::HepRandomEngine by firing once RandFlat "
                                 "---- this must be the first time in SimMuon/RPCDigitizer";
  LogDebug("RPCDigiProducer")
      << "[RPCDigiProducer::produce] to activate the test go in RPCDigiProducer.cc and uncomment the line below";
  // LogDebug ("RPCDigiProducer")<<"[RPCDigiProducer::produce] Fired RandFlat :: "<<CLHEP::RandFlat::shoot(engine);

  edm::Handle<CrossingFrame<PSimHit>> cf;
  // Obsolate code, based on getByLabel
  //  e.getByLabel(mix_, collection_for_XF, cf);
  //New code, based on tokens
  e.getByToken(crossingFrameToken, cf);

  std::unique_ptr<MixCollection<PSimHit>> hits(new MixCollection<PSimHit>(cf.product()));

  // Create empty output
  std::unique_ptr<RPCDigiCollection> pDigis(new RPCDigiCollection());
  std::unique_ptr<RPCDigitizerSimLinks> RPCDigitSimLink(new RPCDigitizerSimLinks());

  // run the digitizer
  theDigitizer->doAction(*hits, *pDigis, *RPCDigitSimLink, engine);

  // store them in the event
  e.put(std::move(pDigis));
  e.put(std::move(RPCDigitSimLink), "RPCDigiSimLink");
}
