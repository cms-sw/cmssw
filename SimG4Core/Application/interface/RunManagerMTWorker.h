#ifndef SimG4Core_Application_RunManagerMTWorker_H
#define SimG4Core_Application_RunManagerMTWorker_H

#include "FWCore/Utilities/interface/EDGetToken.h"

#include "SimG4Core/Generators/interface/Generator.h"
#include "SimG4Core/Notification/interface/SimActivityRegistry.h"
#include "SimDataFormats/Forward/interface/LHCTransportLinkContainer.h"

#include <memory>
#include "boost/shared_ptr.hpp"

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
  class ConsumesCollector;
  class HepMCProduct;
}
class Generator;
class RunManagerMT;

class G4Event;
class G4SimEvent;
class G4Run;
class SimTrackManager;

class RunAction;
class EventAction;
class TrackingAction;
class SteppingAction;

class SimRunInterface;

class SensitiveTkDetector;
class SensitiveCaloDetector;

class SimWatcher;
class SimProducer;

class RunManagerMTWorker {
public:
  RunManagerMTWorker(const edm::ParameterSet& iConfig, edm::ConsumesCollector&& i);
  ~RunManagerMTWorker();

  void beginRun(const RunManagerMT& runManagerMaster, const edm::EventSetup& es);
  void endRun();

  void produce(const edm::Event& inpevt, const edm::EventSetup& es, const RunManagerMT& runManagerMaster);

  void abortEvent();
  void abortRun(bool softAbort=false);

  G4SimEvent * simEvent() { return m_simEvent.get(); }
  SimTrackManager* GetSimTrackManager() { return m_trackManager; }
  void             Connect(RunAction*);
  void             Connect(EventAction*);
  void             Connect(TrackingAction*);
  void             Connect(SteppingAction*);

  std::vector<SensitiveTkDetector*>& sensTkDetectors() {
    return m_sensTkDets; 
  }
  std::vector<SensitiveCaloDetector*>& sensCaloDetectors() {
    return m_sensCaloDets; 
  }
  std::vector<boost::shared_ptr<SimProducer> > producers() {
    return m_producers;
  }

private:
  void initializeThread(const RunManagerMT& runManagerMaster, const edm::EventSetup& es);
  void initializeUserActions();

  void initializeRun();
  void terminateRun();

  G4Event *generateEvent(const edm::Event& inpevt);
  void resetGenParticleId(const edm::Event& inpevt);

  static thread_local bool m_threadInitialized;
  static thread_local bool m_runTerminated;

  Generator m_generator;
  edm::EDGetTokenT<edm::HepMCProduct> m_InToken;
  edm::EDGetTokenT<edm::LHCTransportLinkContainer> m_theLHCTlinkToken;
  const bool m_nonBeam;
  const int m_EvtMgrVerbosity;
  edm::ParameterSet m_pRunAction;
  edm::ParameterSet m_pEventAction;
  edm::ParameterSet m_pStackingAction;
  edm::ParameterSet m_pTrackingAction;
  edm::ParameterSet m_pSteppingAction;
  edm::ParameterSet m_p;

  static thread_local RunAction *m_userRunAction;
  static thread_local SimRunInterface *m_runInterface;
  static thread_local SimActivityRegistry m_registry;
  static thread_local SimTrackManager *m_trackManager;
  static thread_local std::vector<SensitiveTkDetector*> m_sensTkDets;
  static thread_local std::vector<SensitiveCaloDetector*> m_sensCaloDets;

  static thread_local G4Run *m_currentRun;

  std::unique_ptr<G4Event> m_currentEvent;
  std::unique_ptr<G4SimEvent> m_simEvent;

  std::vector<boost::shared_ptr<SimWatcher> > m_watchers;
  std::vector<boost::shared_ptr<SimProducer> > m_producers;
};

#endif
