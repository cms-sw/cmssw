#include "FWCore/PluginManager/interface/PluginManager.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimG4Core/Application/interface/OscarMTProducer.h"
#include "SimG4Core/Application/interface/RunManagerMTInit.h"
#include "SimG4Core/Application/interface/RunManagerMT.h"
#include "SimG4Core/Application/interface/RunManagerMTWorker.h"
#include "SimG4Core/Application/interface/G4SimEvent.h"

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
        explicit StaticRandomEngineSetUnset(CLHEP::HepRandomEngine * engine);
        ~StaticRandomEngineSetUnset();
        CLHEP::HepRandomEngine* getEngine() const;
    private:
        CLHEP::HepRandomEngine* m_currentEngine;
        CLHEP::HepRandomEngine* m_previousEngine;
    };
}

OscarMTProducer::OscarMTProducer(edm::ParameterSet const & p, const edm::ParameterSet *)
{
  // Random number generation not allowed here
  StaticRandomEngineSetUnset random(nullptr);

  m_runManagerWorker.reset(new RunManagerMTWorker(p, consumesCollector()));
}

OscarMTProducer::~OscarMTProducer() 
{ }

std::unique_ptr<edm::ParameterSet> OscarMTProducer::initializeGlobalCache(const edm::ParameterSet& iConfig) {
  // Random number generation not allowed here
  StaticRandomEngineSetUnset random(nullptr);

  return std::unique_ptr<edm::ParameterSet>(new edm::ParameterSet(iConfig));
}

std::shared_ptr<OscarMTMasterThread> OscarMTProducer::globalBeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup, const edm::ParameterSet *iConfig) {
  // Random number generation not allowed here
  StaticRandomEngineSetUnset random(nullptr);

  auto runManager = std::make_shared<RunManagerMTInit>(*iConfig);
  auto masterThread = std::make_shared<OscarMTMasterThread>(runManager, iSetup);
  return masterThread;
}

void OscarMTProducer::globalEndRun(const edm::Run& iRun, const edm::EventSetup& iSetup, const RunContext *iContext) {
  iContext->run()->stopThread();
}

void OscarMTProducer::globalEndJob(edm::ParameterSet *iConfig) {
}


void 
OscarMTProducer::beginRun(const edm::Run & r, const edm::EventSetup & es)
{
  // Random number generation not allowed here
  StaticRandomEngineSetUnset random(nullptr);
  m_runManagerWorker->beginRun(runCache()->runManagerMaster(), es);
}

void 
OscarMTProducer::endRun(const edm::Run&, const edm::EventSetup&)
{
  // Random number generation not allowed here
  StaticRandomEngineSetUnset random(nullptr);
  m_runManagerWorker->endRun();
}

void OscarMTProducer::produce(edm::Event & e, const edm::EventSetup & es)
{
  StaticRandomEngineSetUnset random(e.streamID());

  try {
    m_runManagerWorker->produce(e, es, runCache()->runManagerMaster());
  } catch ( const SimG4Exception& simg4ex ) {
       
    edm::LogInfo("SimG4CoreApplication") << " SimG4Exception caght !" 
					 << simg4ex.what();
       
    m_runManagerWorker->abortEvent();
    throw edm::Exception( edm::errors::EventCorruption );
  }

}

StaticRandomEngineSetUnset::StaticRandomEngineSetUnset(
      edm::StreamID const& streamID)
{
  edm::Service<edm::RandomNumberGenerator> rng;
  if ( ! rng.isAvailable()) {
    throw cms::Exception("Configuration")
      << "The OscarMTProducer module requires the RandomNumberGeneratorService\n"
      "which is not present in the configuration file.  You must add the service\n"
      "in the configuration file if you want to run OscarMTProducer";
  }
  m_currentEngine = &(rng->getEngine(streamID));

  m_previousEngine = G4Random::getTheEngine();
  G4Random::setTheEngine(m_currentEngine);
}

StaticRandomEngineSetUnset::StaticRandomEngineSetUnset(
      CLHEP::HepRandomEngine * engine) 
{
  m_currentEngine = engine;
  m_previousEngine = G4Random::getTheEngine();
  G4Random::setTheEngine(m_currentEngine);
}

StaticRandomEngineSetUnset::~StaticRandomEngineSetUnset() 
{
  G4Random::setTheEngine(m_previousEngine);
}

CLHEP::HepRandomEngine* StaticRandomEngineSetUnset::getEngine() const 
{ 
  return m_currentEngine; 
}

DEFINE_FWK_MODULE(OscarMTProducer);
