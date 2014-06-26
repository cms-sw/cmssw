#ifndef SimG4Core_Application_RunManagerMTWorker_H
#define SimG4Core_Application_RunManagerMTWorker_H

#include "FWCore/Utilities/interface/InputTag.h"

#include "SimG4Core/Generators/interface/Generator.h"
#include "SimG4Core/Notification/interface/SimActivityRegistry.h"

#include <memory>

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
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

class RunManagerMTWorker {
public:
  RunManagerMTWorker(const edm::ParameterSet& iConfig);
  ~RunManagerMTWorker();

  void beginRun(const RunManagerMT& runManagerMaster, const edm::EventSetup& es);

  void produce(const edm::Event& inpevt, const edm::EventSetup& es, const RunManagerMT& runManagerMaster);

  void abortEvent();
  void abortRun(bool softAbort=false);

  G4SimEvent * simEvent() { return m_simEvent.get(); }
  SimTrackManager* GetSimTrackManager() { return m_trackManager; }
  void             Connect(RunAction*);
  void             Connect(EventAction*);
  void             Connect(TrackingAction*);
  void             Connect(SteppingAction*);


private:
  void initializeThread(const RunManagerMT& runManagerMaster, const edm::EventSetup& es);
  void initializeUserActions();

  void initializeRun();
  void terminateRun();

  G4Event *generateEvent(const edm::Event& inpevt);

  static thread_local bool m_threadInitialized;

  Generator m_generator;
  std::string m_InTag;
  const bool m_nonBeam;
  const int m_EvtMgrVerbosity;
  edm::ParameterSet m_pRunAction;
  edm::ParameterSet m_pEventAction;
  edm::ParameterSet m_pStackingAction;
  edm::ParameterSet m_pTrackingAction;
  edm::ParameterSet m_pSteppingAction;
  edm::ParameterSet m_p;

  static thread_local std::vector<SensitiveTkDetector*> m_sensTkDets;
  static thread_local std::vector<SensitiveCaloDetector*> m_sensCaloDets;

  static thread_local RunAction *m_userRunAction;
  static thread_local SimRunInterface *m_runInterface;

  std::unique_ptr<G4Event> m_currentEvent;
  std::unique_ptr<G4SimEvent> m_simEvent;

  static thread_local G4Run *m_currentRun;

  static thread_local SimActivityRegistry m_registry;

  static thread_local SimTrackManager *m_trackManager;
};

#endif
