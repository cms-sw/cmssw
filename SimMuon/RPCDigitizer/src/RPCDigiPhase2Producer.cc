#include "FWCore/AbstractServices/interface/RandomNumberGenerator.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimMuon/RPCDigitizer/src/RPCDigiPhase2Producer.h"
#include "SimMuon/RPCDigitizer/src/RPCDigitizerPhase2.h"
#include "SimMuon/RPCDigitizer/src/RPCSimSetUp.h"
#include "SimMuon/RPCDigitizer/src/RPCSynchronizer.h"

#include "CLHEP/Random/RandFlat.h"

#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace CLHEP {
  class HepRandomEngine;
}

RPCDigiPhase2Producer::RPCDigiPhase2Producer(const edm::ParameterSet& ps) {
  produces<RPCDigiPhase2Collection>();
  produces<RPCDigitizerPhase2SimLinks>("RPCDigiPhase2SimLink");

  //Name of Collection used for create the XF
  const std::string& mix = ps.getParameter<std::string>("mixLabel");
  const std::set<std::string> collections_for_XF_{ps.getParameter<std::string>("inputCollection"),
                                                  ps.getParameter<std::string>("inputCollectionPU")};
  for (const auto& cname : collections_for_XF_) {
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("RPCDigiProducer") << "Creating CrossingFrame Consumers for InputTag " << mix << ":" << cname;
#endif
    crossingFrameTokens_.push_back(consumes<CrossingFrame<PSimHit>>(edm::InputTag(mix, cname)));
  }

  edm::Service<edm::RandomNumberGenerator> rng;
  if (!rng.isAvailable()) {
    throw cms::Exception("Configuration")
        << "RPCDigitizerPhase2 requires the RandomNumberGeneratorService\n"
           "which is not present in the configuration file.  You must add the service\n"
           "in the configuration file or remove the modules that require it.";
  };

  theRPCSimSetUpRPC_ = std::make_unique<RPCSimSetUp>(ps);
  theRPCDigitizerPhase2_ = std::make_unique<RPCDigitizerPhase2>(ps);
  geomToken_ = esConsumes<RPCGeometry, MuonGeometryRecord, edm::Transition::BeginRun>();
  noiseToken_ = esConsumes<RPCStripNoises, RPCStripNoisesRcd, edm::Transition::BeginRun>();
  clsToken_ = esConsumes<RPCClusterSize, RPCClusterSizeRcd, edm::Transition::BeginRun>();
}

RPCDigiPhase2Producer::~RPCDigiPhase2Producer() {}

void RPCDigiPhase2Producer::beginRun(const edm::Run& r, const edm::EventSetup& eventSetup) {
  edm::ESHandle<RPCGeometry> hGeom = eventSetup.getHandle(geomToken_);
  pGeom_ = &*hGeom;

  edm::ESHandle<RPCStripNoises> noiseRcd = eventSetup.getHandle(noiseToken_);
  edm::ESHandle<RPCClusterSize> clsRcd = eventSetup.getHandle(clsToken_);

  //setup the two digi models
  theRPCSimSetUpRPC_->setGeometry(pGeom_);
  theRPCSimSetUpRPC_->setRPCSetUp(noiseRcd->getVNoise(), clsRcd->getCls());

  //setup the two digitizers
  theRPCDigitizerPhase2_->setGeometry(pGeom_);
  theRPCDigitizerPhase2_->setRPCSimSetUp(theRPCSimSetUpRPC_.get());
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

  //New code, based on tokens
  std::vector<const CrossingFrame<PSimHit>*> cf_list;
  for (const auto& token : crossingFrameTokens_) {
    const auto& handle = e.getHandle(token);
    if (handle.isValid()) {
      cf_list.emplace_back(handle.product());
    }
  }
  auto hits = std::make_unique<MixCollection<PSimHit>>(cf_list);

  // Create empty output
  auto pDigis = std::make_unique<RPCDigiPhase2Collection>();
  auto RPCDigitSimLink = std::make_unique<RPCDigitizerPhase2SimLinks>();

  theRPCDigitizerPhase2_->doAction(
      *hits, *pDigis, *RPCDigitSimLink, engine);  //make "bakelite RPC" digitizer do the action

  e.put(std::move(pDigis));
  //store the SimDigiLinks in the event
  e.put(std::move(RPCDigitSimLink), "RPCDigiPhase2SimLink");
}
