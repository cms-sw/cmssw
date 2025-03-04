#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "SimMuon/RPCDigitizer/src/RPCSimSetUp.h"
#include "SimMuon/RPCDigitizer/src/RPCDigiPhase2Producer.h"
#include "SimMuon/RPCDigitizer/src/RPCDigitizerPhase2.h"
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

RPCDigiPhase2Producer::RPCDigiPhase2Producer(const edm::ParameterSet& ps) {
  produces<RPCDigiPhase2Collection>();
  produces<RPCDigitizerPhase2SimLinks>("RPCDigiPhase2SimLink");

  //Name of Collection used for create the XF
  mix_ = ps.getParameter<std::string>("mixLabel");
  collection_for_XF = ps.getParameter<std::string>("InputCollection");

  edm::Service<edm::RandomNumberGenerator> rng;
  if (!rng.isAvailable()) {
    throw cms::Exception("Configuration")
        << "RPCDigitizerPhase2 requires the RandomNumberGeneratorService\n"
           "which is not present in the configuration file.  You must add the service\n"
           "in the configuration file or remove the modules that require it.";
  };

  theRPCSimSetUpRPC = new RPCSimSetUp(ps);
  theRPCDigitizerPhase2 = new RPCDigitizerPhase2(ps);
  crossingFrameToken = consumes<CrossingFrame<PSimHit>>(edm::InputTag(mix_, collection_for_XF));
  geomToken = esConsumes<RPCGeometry, MuonGeometryRecord, edm::Transition::BeginRun>();
  noiseToken = esConsumes<RPCStripNoises, RPCStripNoisesRcd, edm::Transition::BeginRun>();
  clsToken = esConsumes<RPCClusterSize, RPCClusterSizeRcd, edm::Transition::BeginRun>();
}

RPCDigiPhase2Producer::~RPCDigiPhase2Producer() {
  delete theRPCDigitizerPhase2;
  delete theRPCSimSetUpRPC;
}

void RPCDigiPhase2Producer::beginRun(const edm::Run& r, const edm::EventSetup& eventSetup) {
  edm::ESHandle<RPCGeometry> hGeom = eventSetup.getHandle(geomToken);
  const RPCGeometry* pGeom = &*hGeom;
  _pGeom = &*hGeom;
  edm::ESHandle<RPCStripNoises> noiseRcd = eventSetup.getHandle(noiseToken);

  edm::ESHandle<RPCClusterSize> clsRcd = eventSetup.getHandle(clsToken);

  //setup the two digi models
  theRPCSimSetUpRPC->setGeometry(pGeom);
  theRPCSimSetUpRPC->setRPCSetUp(noiseRcd->getVNoise(), clsRcd->getCls());

  //setup the two digitizers
  theRPCDigitizerPhase2->setGeometry(pGeom);
  theRPCDigitizerPhase2->setRPCSimSetUp(theRPCSimSetUpRPC);
}

void RPCDigiPhase2Producer::produce(edm::Event& e, const edm::EventSetup& eventSetup) {
  edm::Service<edm::RandomNumberGenerator> rng;
  CLHEP::HepRandomEngine* engine = &rng->getEngine(e.streamID());

  LogDebug("RPCDigiPhase2Producer") << "[RPCDigiPhase2Producer::produce] got the CLHEP::HepRandomEngine engine from "
                                       "the edm::Event.streamID() and edm::Service<edm::RandomNumberGenerator>";
  LogDebug("RPCDigiPhase2Producer") << "[RPCDigiPhase2Producer::produce] test the CLHEP::HepRandomEngine by firing "
                                       "once RandFlat ---- this must be the first time in SimMuon/RPCDigitizerPhase2";
  LogDebug("RPCDigiPhase2Producer") << "[RPCDigiPhase2Producer::produce] to activate the test go in "
                                       "RPCDigiPhase2Producer.cc and uncomment the line below";

  edm::Handle<CrossingFrame<PSimHit>> cf;
  e.getByToken(crossingFrameToken, cf);

  std::unique_ptr<MixCollection<PSimHit>> hits(new MixCollection<PSimHit>(cf.product()));

  // Create empty output
  std::unique_ptr<RPCDigiPhase2Collection> pDigis(new RPCDigiPhase2Collection());
  std::unique_ptr<RPCDigitizerPhase2SimLinks> RPCDigitSimLink(new RPCDigitizerPhase2SimLinks());

  theRPCDigitizerPhase2->doAction(
      *hits, *pDigis, *RPCDigitSimLink, engine);  //make "bakelite RPC" digitizer do the action

  e.put(std::move(pDigis));
  //store the SimDigiLinks in the event
  e.put(std::move(RPCDigitSimLink), "RPCDigiPhase2SimLink");
}
