#ifndef SimG4Core_Application_RunManagerMTWorker_H
#define SimG4Core_Application_RunManagerMTWorker_H

#include "FWCore/Utilities/interface/EDGetToken.h"
#include "DataFormats/Provenance/interface/RunID.h"

#include "SimG4Core/Generators/interface/Generator.h"
#include "SimDataFormats/Forward/interface/LHCTransportLinkContainer.h"

#include <memory>

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
class CMSSteppingVerbose;

class SensitiveTkDetector;
class SensitiveCaloDetector;

class SimWatcher;
class SimProducer;

class RunManagerMTWorker {
public:
  explicit RunManagerMTWorker(const edm::ParameterSet& iConfig, edm::ConsumesCollector&& i);
  ~RunManagerMTWorker();

  void endRun();

  void produce(const edm::Event& inpevt, const edm::EventSetup& es, RunManagerMT& runManagerMaster);

  void abortEvent();
  void abortRun(bool softAbort=false);

  inline G4SimEvent * simEvent() { return m_simEvent.get(); }

  void Connect(RunAction*);
  void Connect(EventAction*);
  void Connect(TrackingAction*);
  void Connect(SteppingAction*);

  SimTrackManager* GetSimTrackManager();
  std::vector<SensitiveTkDetector*>& sensTkDetectors();
  std::vector<SensitiveCaloDetector*>& sensCaloDetectors();
  std::vector<std::shared_ptr<SimProducer> > producers();

private:

  void initializeTLS();
  void initializeThread(RunManagerMT& runManagerMaster, const edm::EventSetup& es);
  void initializeUserActions();

  void initializeRun();
  void terminateRun();

  G4Event *generateEvent(const edm::Event& inpevt);
  void resetGenParticleId(const edm::Event& inpevt);

  Generator m_generator;
  edm::EDGetTokenT<edm::HepMCProduct> m_InToken;
  edm::EDGetTokenT<edm::LHCTransportLinkContainer> m_theLHCTlinkToken;

  bool m_nonBeam;
  bool m_pUseMagneticField;
  bool m_hasWatchers;
  int  m_EvtMgrVerbosity;

  edm::ParameterSet m_pField;
  edm::ParameterSet m_pRunAction;
  edm::ParameterSet m_pEventAction;
  edm::ParameterSet m_pStackingAction;
  edm::ParameterSet m_pTrackingAction;
  edm::ParameterSet m_pSteppingAction;
  edm::ParameterSet m_pCustomUIsession;
  edm::ParameterSet m_p;

  struct TLSData;
  static thread_local TLSData *m_tls;

  std::unique_ptr<G4SimEvent> m_simEvent;
  std::unique_ptr<CMSSteppingVerbose> m_sVerbose;
};

#endif
