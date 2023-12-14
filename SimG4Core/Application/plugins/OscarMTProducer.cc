#include <iostream>
#include <memory>

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"

#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "SimG4Core/Application/interface/OscarMTMasterThread.h"
#include "SimG4Core/Application/interface/RunManagerMT.h"
#include "SimG4Core/Application/interface/RunManagerMTWorker.h"
#include "SimG4Core/Notification/interface/TmpSimEvent.h"
#include "SimG4Core/Notification/interface/TmpSimVertex.h"

#include "SimG4Core/SensitiveDetector/interface/SensitiveTkDetector.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveCaloDetector.h"

#include "SimG4Core/Watcher/interface/SimProducer.h"

#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"

#include "SimG4Core/Application/interface/ThreadHandoff.h"

#include "Randomize.hh"

// for some reason void doesn't compile
class OscarMTProducer : public edm::stream::EDProducer<edm::GlobalCache<OscarMTMasterThread>, edm::RunCache<int>> {
public:
  typedef std::vector<std::shared_ptr<SimProducer>> Producers;

  explicit OscarMTProducer(edm::ParameterSet const& p, const OscarMTMasterThread*);
  ~OscarMTProducer() override;

  static std::unique_ptr<OscarMTMasterThread> initializeGlobalCache(const edm::ParameterSet& iConfig);
  static std::shared_ptr<int> globalBeginRun(const edm::Run& iRun,
                                             const edm::EventSetup& iSetup,
                                             const OscarMTMasterThread* masterThread);
  static void globalEndRun(const edm::Run& iRun, const edm::EventSetup& iSetup, const RunContext* iContext);
  static void globalEndJob(OscarMTMasterThread* masterThread);

  void beginRun(const edm::Run& r, const edm::EventSetup& c) override;
  void endRun(const edm::Run& r, const edm::EventSetup& c) override;
  void produce(edm::Event& e, const edm::EventSetup& c) override;

private:
  omt::ThreadHandoff m_handoff;
  std::unique_ptr<RunManagerMTWorker> m_runManagerWorker;
  const OscarMTMasterThread* m_masterThread;
  const edm::ParameterSetID m_psetID;
  int m_verbose;
  CMS_SA_ALLOW static const OscarMTMasterThread* s_masterThread;
  CMS_SA_ALLOW static edm::ParameterSetID s_psetID;
};

const OscarMTMasterThread* OscarMTProducer::s_masterThread = nullptr;
edm::ParameterSetID OscarMTProducer::s_psetID{};

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

    CLHEP::HepRandomEngine* currentEngine() { return m_currentEngine; }

  private:
    CLHEP::HepRandomEngine* m_currentEngine;
    CLHEP::HepRandomEngine* m_previousEngine;
  };
}  // namespace

OscarMTProducer::OscarMTProducer(edm::ParameterSet const& p, const OscarMTMasterThread* ms)
    : m_handoff{p.getUntrackedParameter<int>("workerThreadStackSize", 10 * 1024 * 1024)}, m_psetID{p.id()} {
  m_verbose = p.getParameter<int>("EventVerbose");
  // Random number generation not allowed here
  StaticRandomEngineSetUnset random(nullptr);

  auto token = edm::ServiceRegistry::instance().presentToken();
  m_handoff.runAndWait([this, &p, token]() {
    edm::ServiceRegistry::Operate guard{token};
    StaticRandomEngineSetUnset random(nullptr);
    m_runManagerWorker = std::make_unique<RunManagerMTWorker>(p, consumesCollector());
  });
  m_masterThread = (nullptr != ms) ? ms : s_masterThread;
  assert(m_masterThread);
  m_masterThread->callConsumes(consumesCollector());

  // declair hit collections
  produces<edm::SimTrackContainer>().setBranchAlias("SimTracks");
  produces<edm::SimVertexContainer>().setBranchAlias("SimVertices");

  auto trackHits = p.getParameter<std::vector<std::string>>("TrackHits");
  for (auto const& ss : trackHits) {
    produces<edm::PSimHitContainer>(ss);
  }

  auto caloHits = p.getParameter<std::vector<std::string>>("CaloHits");
  for (auto const& ss : caloHits) {
    produces<edm::PCaloHitContainer>(ss);
  }

  //register any products
  auto& producers = m_runManagerWorker->producers();
  for (auto& ptr : producers) {
    ptr->registerProducts(producesCollector());
  }
  edm::LogVerbatim("SimG4CoreApplication")
      << "OscarMTProducer is constructed with hit collections:" << trackHits.size() << " tracking type; "
      << caloHits.size() << " calo type; " << producers.size() << " watcher type.";
}

OscarMTProducer::~OscarMTProducer() {
  auto token = edm::ServiceRegistry::instance().presentToken();
  m_handoff.runAndWait([this, token]() {
    edm::ServiceRegistry::Operate guard{token};
    m_runManagerWorker.reset();
  });
}

std::unique_ptr<OscarMTMasterThread> OscarMTProducer::initializeGlobalCache(const edm::ParameterSet& iConfig) {
  // Random number generation not allowed here
  StaticRandomEngineSetUnset random(nullptr);
  edm::LogVerbatim("SimG4CoreApplication") << "OscarMTProducer::initializeGlobalCache";
  if (nullptr == s_masterThread) {
    auto ret = std::make_unique<OscarMTMasterThread>(iConfig);
    s_masterThread = ret.get();
    s_psetID = iConfig.id();
    return ret;
  }
  return {};
}

std::shared_ptr<int> OscarMTProducer::globalBeginRun(const edm::Run&,
                                                     const edm::EventSetup& iSetup,
                                                     const OscarMTMasterThread* masterThread) {
  // Random number generation not allowed here
  StaticRandomEngineSetUnset random(nullptr);
  edm::LogVerbatim("SimG4CoreApplication") << "OscarMTProducer::globalBeginRun";
  if (masterThread) {
    masterThread->beginRun(iSetup);
  }
  edm::LogVerbatim("SimG4CoreApplication") << "OscarMTProducer::globalBeginRun done";
  return std::shared_ptr<int>();
}

void OscarMTProducer::globalEndRun(const edm::Run&, const edm::EventSetup&, const RunContext* iContext) {
  edm::LogVerbatim("SimG4CoreApplication") << "OscarMTProducer::globalEndRun";
  if (nullptr != iContext->global()) {
    iContext->global()->endRun();
  }
}

void OscarMTProducer::globalEndJob(OscarMTMasterThread* masterThread) {
  edm::LogVerbatim("SimG4CoreApplication") << "OscarMTProducer::globalEndJob";
  if (masterThread) {
    masterThread->stopThread();
  }
}

void OscarMTProducer::beginRun(const edm::Run&, const edm::EventSetup& es) {
  if (s_psetID != m_psetID) {
    throw cms::Exception("DiffOscarMTProducers")
        << "At least two different OscarMTProducer instances have been"
           "loaded into the job and they have different configurations.\n"
           " All OscarMTProducers in a job must have exactly the same configuration.";
  }
  int id = m_runManagerWorker->getThreadIndex();
  edm::LogVerbatim("SimG4CoreApplication") << "OscarMTProducer::beginRun threadID=" << id;
  auto token = edm::ServiceRegistry::instance().presentToken();
  m_handoff.runAndWait([this, &es, token]() {
    edm::ServiceRegistry::Operate guard{token};
    m_runManagerWorker->beginRun(es);
    m_runManagerWorker->initializeG4(m_masterThread->runManagerMasterPtr(), es);
  });
  edm::LogVerbatim("SimG4CoreApplication") << "OscarMTProducer::beginRun done threadID=" << id;
}

void OscarMTProducer::endRun(const edm::Run&, const edm::EventSetup&) {
  StaticRandomEngineSetUnset random(nullptr);
  int id = m_runManagerWorker->getThreadIndex();
  edm::LogVerbatim("SimG4CoreApplication") << "OscarMTProducer::endRun threadID=" << id;
  auto token = edm::ServiceRegistry::instance().presentToken();
  m_handoff.runAndWait([this, token]() {
    edm::ServiceRegistry::Operate guard{token};
    m_runManagerWorker->endRun();
  });
  edm::LogVerbatim("SimG4CoreApplication") << "OscarMTProducer::endRun done threadID=" << id;
}

void OscarMTProducer::produce(edm::Event& e, const edm::EventSetup& es) {
  StaticRandomEngineSetUnset random(e.streamID());
  auto engine = random.currentEngine();
  int id = m_runManagerWorker->getThreadIndex();
  if (0 < m_verbose) {
    edm::LogVerbatim("SimG4CoreApplication")
        << "Produce event " << e.id() << " stream " << e.streamID() << " threadID=" << id;
    //edm::LogVerbatim("SimG4CoreApplication") << " rand= " << G4UniformRand();
  }

  auto& sTk = m_runManagerWorker->sensTkDetectors();
  auto& sCalo = m_runManagerWorker->sensCaloDetectors();

  TmpSimEvent* evt = nullptr;
  auto token = edm::ServiceRegistry::instance().presentToken();
  m_handoff.runAndWait([this, &e, &es, &evt, token, engine]() {
    edm::ServiceRegistry::Operate guard{token};
    StaticRandomEngineSetUnset random(engine);
    evt = m_runManagerWorker->produce(e, es, m_masterThread->runManagerMaster());
  });

  std::unique_ptr<edm::SimTrackContainer> p1(new edm::SimTrackContainer);
  std::unique_ptr<edm::SimVertexContainer> p2(new edm::SimVertexContainer);
  evt->load(*p1);
  evt->load(*p2);

  if (0 < m_verbose) {
    edm::LogVerbatim("SimG4CoreApplication") << "Produced " << p2->size() << " SimVertex objects";
    if (1 < m_verbose) {
      int nn = p2->size();
      for (int i = 0; i < nn; ++i) {
        edm::LogVerbatim("Vertex") << " " << (*p2)[i] << " " << (*p2)[i].processType();
      }
    }
    edm::LogVerbatim("SimG4CoreApplication") << "Produced " << p1->size() << " SimTrack objects";
    if (1 < m_verbose) {
      int nn = p1->size();
      for (int i = 0; i < nn; ++i) {
        edm::LogVerbatim("Track") << " " << i << ". " << (*p1)[i] << " " << (*p1)[i].crossedBoundary() << " "
                                  << (*p1)[i].getIDAtBoundary();
      }
    }
  }
  e.put(std::move(p1));
  e.put(std::move(p2));

  for (auto const& tracker : sTk) {
    const std::vector<std::string>& v = tracker->getNames();
    for (auto const& name : v) {
      std::unique_ptr<edm::PSimHitContainer> product(new edm::PSimHitContainer);
      tracker->fillHits(*product, name);
      if (0 < m_verbose && product != nullptr && !product->empty())
        edm::LogVerbatim("SimG4CoreApplication") << "Produced " << product->size() << " tracker hits <" << name << ">";
      e.put(std::move(product), name);
    }
  }
  for (auto const& calo : sCalo) {
    const std::vector<std::string>& v = calo->getNames();
    for (auto const& name : v) {
      std::unique_ptr<edm::PCaloHitContainer> product(new edm::PCaloHitContainer);
      calo->fillHits(*product, name);
      if (0 < m_verbose && product != nullptr && !product->empty())
        edm::LogVerbatim("SimG4CoreApplication") << "Produced " << product->size() << " calo hits <" << name << ">";
      e.put(std::move(product), name);
    }
  }

  auto& producers = m_runManagerWorker->producers();
  for (auto& prod : producers) {
    prod.get()->produce(e, es);
  }
  if (0 < m_verbose) {
    edm::LogVerbatim("SimG4CoreApplication")
        << "Event is produced event " << e.id() << " streamID=" << e.streamID() << " threadID=" << id;
    //edm::LogWarning("SimG4CoreApplication") << "EventID=" << e.id() << " rand=" << G4UniformRand();
  }
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
