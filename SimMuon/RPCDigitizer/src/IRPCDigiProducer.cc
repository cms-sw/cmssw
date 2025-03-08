#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "SimMuon/RPCDigitizer/src/RPCSimSetUp.h"
#include "SimMuon/RPCDigitizer/src/IRPCDigiProducer.h"
#include "SimMuon/RPCDigitizer/src/IRPCDigitizer.h"
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

IRPCDigiProducer::IRPCDigiProducer(const edm::ParameterSet& ps) {
  produces<IRPCDigiCollection>();
  produces<IRPCDigitizerSimLinks>("IRPCDigiSimLink");

  //Name of Collection used for create the XF
  mix_ = ps.getParameter<std::string>("mixLabel");
  collection_for_XF = ps.getParameter<std::string>("InputCollection");

  edm::Service<edm::RandomNumberGenerator> rng;
  if (!rng.isAvailable()) {
    throw cms::Exception("Configuration")
        << "RPCDigitizer requires the RandomNumberGeneratorService\n"
           "which is not present in the configuration file.  You must add the service\n"
           "in the configuration file or remove the modules that require it.";
  };

  theRPCSimSetUpIRPC = new RPCSimSetUp(ps);
  theIRPCDigitizer = new IRPCDigitizer(ps);
  crossingFrameToken = consumes<CrossingFrame<PSimHit>>(edm::InputTag(mix_, collection_for_XF));
  geomToken = esConsumes<RPCGeometry, MuonGeometryRecord, edm::Transition::BeginRun>();
  noiseToken = esConsumes<RPCStripNoises, RPCStripNoisesRcd, edm::Transition::BeginRun>();
  clsToken = esConsumes<RPCClusterSize, RPCClusterSizeRcd, edm::Transition::BeginRun>();
}

IRPCDigiProducer::~IRPCDigiProducer() {
  delete theIRPCDigitizer;
  delete theRPCSimSetUpIRPC;
}

void IRPCDigiProducer::beginRun(const edm::Run& r, const edm::EventSetup& eventSetup) {
  edm::ESHandle<RPCGeometry> hGeom = eventSetup.getHandle(geomToken);
  const RPCGeometry* pGeom = &*hGeom;
  _pGeom = &*hGeom;

  edm::ESHandle<RPCStripNoises> noiseRcd = eventSetup.getHandle(noiseToken);

  edm::ESHandle<RPCClusterSize> clsRcd = eventSetup.getHandle(clsToken);
  //eventSetup.get<RPCClusterSizeRcd>().get(clsRcd);

  //setup the two digi models
  theRPCSimSetUpIRPC->setGeometry(pGeom);
  theRPCSimSetUpIRPC->setRPCSetUp(noiseRcd->getVNoise(), clsRcd->getCls());

  //setup the two digitizers
  theIRPCDigitizer->setGeometry(pGeom);
  theIRPCDigitizer->setRPCSimSetUp(theRPCSimSetUpIRPC);
}

void IRPCDigiProducer::produce(edm::Event& e, const edm::EventSetup& eventSetup) {
  edm::Service<edm::RandomNumberGenerator> rng;
  CLHEP::HepRandomEngine* engine = &rng->getEngine(e.streamID());

  LogDebug("IRPCDigiProducer") << "[IRPCDigiProducer::produce] got the CLHEP::HepRandomEngine engine from "
                                  "the edm::Event.streamID() and edm::Service<edm::RandomNumberGenerator>";
  LogDebug("IRPCDigiProducer") << "[IRPCDigiProducer::produce] test the CLHEP::HepRandomEngine by firing "
                                  "once RandFlat ---- this must be the first time in SimMuon/RPCDigitizer";
  LogDebug("IRPCDigiProducer") << "[IRPCDigiProducer::produce] to activate the test go in "
                                  "IRPCDigiProducer.cc and uncomment the line below";

  const edm::Handle<CrossingFrame<PSimHit>>& cf = e.getHandle(crossingFrameToken);

  std::unique_ptr<MixCollection<PSimHit>> hits(new MixCollection<PSimHit>(cf.product()));

  // Create empty output
  std::unique_ptr<IRPCDigiCollection> pDigis(new IRPCDigiCollection());
  std::unique_ptr<IRPCDigitizerSimLinks> IRPCDigitSimLink(new IRPCDigitizerSimLinks());

  theIRPCDigitizer->doAction(*hits, *pDigis, *IRPCDigitSimLink, engine);  //make "IRPC" digitizer do the action

  e.put(std::move(pDigis));
  //store the SimDigiLinks in the event
  e.put(std::move(IRPCDigitSimLink), "IRPCDigiSimLink");
}
