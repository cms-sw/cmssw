#include "FWCore/PluginManager/interface/PluginManager.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "OscarMTProducer.h"

#include "SimG4Core/Application/interface/RunManagerMT.h"
#include "SimG4Core/Application/interface/RunManagerMTWorker.h"
#include "SimG4Core/Notification/interface/G4SimEvent.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveTkDetector.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveCaloDetector.h"

#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"

#include "SimG4Core/Watcher/interface/SimProducer.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "SimG4Core/Notification/interface/SimG4Exception.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "Randomize.hh"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>

namespace edm {
  class StreamID;
}

namespace {
  //
  // this machinery allows to set CLHEP static engine
  // to the one defined by RandomNumberGenerator service
  // at the beginning of an event, and reset it back to
  // "default-default" at the end of the event;
  // Dave D. has decided to implement it this way because
  // we don't know if there're other modules using CLHEP
  // static engine, thus we want to ensure that the one
  // we use for OscarMTProducer is unique to OscarMTProducer
  //
  // !!! This not only sets the random engine used by GEANT.
  // There are a few SimWatchers/SimProducers that generate
  // random number and also use the global CLHEP random engine
  // set by this code. If we ever change this design be careful
  // not to forget about them!!!

  class StaticRandomEngineSetUnset {
  public:
    StaticRandomEngineSetUnset(edm::StreamID const&);
    explicit StaticRandomEngineSetUnset(CLHEP::HepRandomEngine* engine);
    ~StaticRandomEngineSetUnset();

  private:
    CLHEP::HepRandomEngine* m_currentEngine;
    CLHEP::HepRandomEngine* m_previousEngine;
  };
}  // namespace

OscarMTProducer::OscarMTProducer(edm::ParameterSet const& p, const OscarMTMasterThread* ms) {
  // Random number generation not allowed here
  StaticRandomEngineSetUnset random(nullptr);

  m_runManagerWorker = std::make_unique<RunManagerMTWorker>(p, consumesCollector());
  m_masterThread = ms;

  produces<edm::SimTrackContainer>().setBranchAlias("SimTracks");
  produces<edm::SimVertexContainer>().setBranchAlias("SimVertices");
  produces<edm::PSimHitContainer>("TrackerHitsPixelBarrelLowTof");
  produces<edm::PSimHitContainer>("TrackerHitsPixelBarrelHighTof");
  produces<edm::PSimHitContainer>("TrackerHitsTIBLowTof");
  produces<edm::PSimHitContainer>("TrackerHitsTIBHighTof");
  produces<edm::PSimHitContainer>("TrackerHitsTIDLowTof");
  produces<edm::PSimHitContainer>("TrackerHitsTIDHighTof");
  produces<edm::PSimHitContainer>("TrackerHitsPixelEndcapLowTof");
  produces<edm::PSimHitContainer>("TrackerHitsPixelEndcapHighTof");
  produces<edm::PSimHitContainer>("TrackerHitsTOBLowTof");
  produces<edm::PSimHitContainer>("TrackerHitsTOBHighTof");
  produces<edm::PSimHitContainer>("TrackerHitsTECLowTof");
  produces<edm::PSimHitContainer>("TrackerHitsTECHighTof");

  produces<edm::PSimHitContainer>("TotemHitsT1");
  produces<edm::PSimHitContainer>("TotemHitsT2Gem");
  produces<edm::PSimHitContainer>("TotemHitsRP");
  produces<edm::PSimHitContainer>("CTPPSPixelHits");
  produces<edm::PSimHitContainer>("CTPPSTimingHits");
  produces<edm::PSimHitContainer>("FP420SI");
  produces<edm::PSimHitContainer>("BSCHits");
  produces<edm::PSimHitContainer>("PLTHits");
  produces<edm::PSimHitContainer>("BCM1FHits");
  produces<edm::PSimHitContainer>("BHMHits");
  produces<edm::PSimHitContainer>("FastTimerHitsBarrel");
  produces<edm::PSimHitContainer>("FastTimerHitsEndcap");

  produces<edm::PCaloHitContainer>("EcalHitsEB");
  produces<edm::PCaloHitContainer>("EcalHitsEE");
  produces<edm::PCaloHitContainer>("EcalHitsES");
  produces<edm::PCaloHitContainer>("HcalHits");
  produces<edm::PCaloHitContainer>("CaloHitsTk");
  produces<edm::PCaloHitContainer>("HGCHitsEE");
  produces<edm::PCaloHitContainer>("HGCHitsHEfront");
  produces<edm::PCaloHitContainer>("HGCHitsHEback");

  produces<edm::PSimHitContainer>("MuonDTHits");
  produces<edm::PSimHitContainer>("MuonCSCHits");
  produces<edm::PSimHitContainer>("MuonRPCHits");
  produces<edm::PSimHitContainer>("MuonGEMHits");
  produces<edm::PSimHitContainer>("MuonME0Hits");
  produces<edm::PCaloHitContainer>("CastorPL");
  produces<edm::PCaloHitContainer>("CastorFI");
  produces<edm::PCaloHitContainer>("CastorBU");
  produces<edm::PCaloHitContainer>("CastorTU");
  produces<edm::PCaloHitContainer>("EcalTBH4BeamHits");
  produces<edm::PCaloHitContainer>("HcalTB06BeamHits");
  produces<edm::PCaloHitContainer>("ZDCHITS");
  produces<edm::PCaloHitContainer>("ChamberHits");
  produces<edm::PCaloHitContainer>("FibreHits");
  produces<edm::PCaloHitContainer>("WedgeHits");
  produces<edm::PCaloHitContainer>("HFNoseHits");
  produces<edm::PCaloHitContainer>("TotemHitsT2Scint");

  //register any products
  auto& producers = m_runManagerWorker->producers();

  for (Producers::iterator itProd = producers.begin(); itProd != producers.end(); ++itProd) {
    (*itProd)->registerProducts(producesCollector());
  }
  edm::LogVerbatim("SimG4CoreApplication") << "OscarMTProducer is constructed";
}

OscarMTProducer::~OscarMTProducer() {}

std::unique_ptr<OscarMTMasterThread> OscarMTProducer::initializeGlobalCache(const edm::ParameterSet& iConfig) {
  // Random number generation not allowed here
  StaticRandomEngineSetUnset random(nullptr);
  edm::LogVerbatim("SimG4CoreApplication") << "OscarMTProducer::initializeGlobalCache";

  return std::unique_ptr<OscarMTMasterThread>(new OscarMTMasterThread(iConfig));
}

std::shared_ptr<int> OscarMTProducer::globalBeginRun(const edm::Run& iRun,
                                                     const edm::EventSetup& iSetup,
                                                     const OscarMTMasterThread* masterThread) {
  // Random number generation not allowed here
  StaticRandomEngineSetUnset random(nullptr);

  edm::LogVerbatim("SimG4CoreApplication") << "OscarMTProducer::globalBeginRun";
  masterThread->beginRun(iSetup);
  edm::LogVerbatim("SimG4CoreApplication") << "OscarMTProducer::globalBeginRun done";
  return std::shared_ptr<int>();
}

void OscarMTProducer::globalEndRun(const edm::Run& iRun, const edm::EventSetup& iSetup, const RunContext* iContext) {
  edm::LogVerbatim("SimG4CoreApplication") << "OscarMTProducer::globalEndRun";
  iContext->global()->endRun();
}

void OscarMTProducer::globalEndJob(OscarMTMasterThread* masterThread) {
  edm::LogVerbatim("SimG4CoreApplication") << "OscarMTProducer::globalEndJob";
  masterThread->stopThread();
}

void OscarMTProducer::beginRun(const edm::Run&, const edm::EventSetup& es) {
  edm::LogVerbatim("SimG4CoreApplication") << "OscarMTProducer::beginRun";
  m_runManagerWorker->initializeG4(m_masterThread->runManagerMasterPtr(), es);
  edm::LogVerbatim("SimG4CoreApplication") << "OscarMTProducer::beginRun done";
}

void OscarMTProducer::endRun(const edm::Run&, const edm::EventSetup&) {
  // Random number generation not allowed here
  StaticRandomEngineSetUnset random(nullptr);
  edm::LogVerbatim("SimG4CoreApplication") << "OscarMTProducer::endRun";
  m_runManagerWorker->endRun();
  edm::LogVerbatim("SimG4CoreApplication") << "OscarMTProducer::endRun done";
}

void OscarMTProducer::produce(edm::Event& e, const edm::EventSetup& es) {
  StaticRandomEngineSetUnset random(e.streamID());
  edm::LogVerbatim("SimG4CoreApplication") << "Produce event " << e.id() << " stream " << e.streamID();
  LogDebug("SimG4CoreApplication") << "Before event rand= " << G4UniformRand();

  auto& sTk = m_runManagerWorker->sensTkDetectors();
  auto& sCalo = m_runManagerWorker->sensCaloDetectors();

  std::unique_ptr<G4SimEvent> evt;
  try {
    evt = m_runManagerWorker->produce(e, es, globalCache()->runManagerMaster());
  } catch (const SimG4Exception& simg4ex) {
    edm::LogWarning("SimG4CoreApplication") << "SimG4Exception caght! " << simg4ex.what();

    throw edm::Exception(edm::errors::EventCorruption)
        << "SimG4CoreApplication exception in generation of event " << e.id() << " in stream " << e.streamID() << " \n"
        << simg4ex.what();
  }

  std::unique_ptr<edm::SimTrackContainer> p1(new edm::SimTrackContainer);
  std::unique_ptr<edm::SimVertexContainer> p2(new edm::SimVertexContainer);
  evt->load(*p1);
  evt->load(*p2);

  e.put(std::move(p1));
  e.put(std::move(p2));

  for (auto& tracker : sTk) {
    const std::vector<std::string>& v = tracker->getNames();
    for (auto& name : v) {
      std::unique_ptr<edm::PSimHitContainer> product(new edm::PSimHitContainer);
      tracker->fillHits(*product, name);
      if (product != nullptr && !product->empty())
        edm::LogVerbatim("SimG4CoreApplication") << "Produced " << product->size() << " traker hits <" << name << ">";
      e.put(std::move(product), name);
    }
  }
  for (auto& calo : sCalo) {
    const std::vector<std::string>& v = calo->getNames();

    for (auto& name : v) {
      std::unique_ptr<edm::PCaloHitContainer> product(new edm::PCaloHitContainer);
      calo->fillHits(*product, name);
      if (product != nullptr && !product->empty())
        edm::LogVerbatim("SimG4CoreApplication") << "Produced " << product->size() << " calo hits <" << name << ">";
      e.put(std::move(product), name);
    }
  }

  auto& producers = m_runManagerWorker->producers();
  for (auto& prod : producers) {
    prod.get()->produce(e, es);
  }
  edm::LogVerbatim("SimG4CoreApplication") << "Event is produced " << e.id() << " stream " << e.streamID();
  LogDebug("SimG4CoreApplication") << "End of event rand= " << G4UniformRand();
}

StaticRandomEngineSetUnset::StaticRandomEngineSetUnset(edm::StreamID const& streamID) {
  edm::Service<edm::RandomNumberGenerator> rng;
  if (!rng.isAvailable()) {
    throw cms::Exception("Configuration")
        << "The OscarMTProducer module requires the RandomNumberGeneratorService\n"
           "which is not present in the configuration file.  You must add the service\n"
           "in the configuration file if you want to run OscarMTProducer";
  }
  m_currentEngine = &(rng->getEngine(streamID));

  m_previousEngine = G4Random::getTheEngine();
  G4Random::setTheEngine(m_currentEngine);
}

StaticRandomEngineSetUnset::StaticRandomEngineSetUnset(CLHEP::HepRandomEngine* engine) {
  m_currentEngine = engine;
  m_previousEngine = G4Random::getTheEngine();
  G4Random::setTheEngine(m_currentEngine);
}

StaticRandomEngineSetUnset::~StaticRandomEngineSetUnset() { G4Random::setTheEngine(m_previousEngine); }

DEFINE_FWK_MODULE(OscarMTProducer);
