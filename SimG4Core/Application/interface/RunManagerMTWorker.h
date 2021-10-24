#ifndef SimG4Core_Application_RunManagerMTWorker_H
#define SimG4Core_Application_RunManagerMTWorker_H

#include "FWCore/Utilities/interface/EDGetToken.h"
#include "DataFormats/Provenance/interface/RunID.h"

#include "SimG4Core/Generators/interface/Generator.h"
#include "SimDataFormats/Forward/interface/LHCTransportLinkContainer.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include <memory>
#include <tbb/concurrent_vector.h>
#include <unordered_map>
#include <string>

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
  class ConsumesCollector;
  class HepMCProduct;
}  // namespace edm

class Generator;
class RunManagerMT;

class G4RunManagerKernel;
class G4Event;
class G4SimEvent;
class G4Run;
class SimTrackManager;

class RunAction;
class EventAction;
class TrackingAction;
class SteppingAction;
class CMSSteppingVerbose;
class G4Field;

class SensitiveTkDetector;
class SensitiveCaloDetector;
class SensitiveDetectorMakerBase;

class SimWatcher;
class SimRunInterface;
class SimActivityRegistry;
class SimTrackManager;

class RunManagerMTWorker {
public:
  explicit RunManagerMTWorker(const edm::ParameterSet& iConfig, edm::ConsumesCollector&& iC);
  ~RunManagerMTWorker();

  void beginRun(const edm::EventSetup&);
  void endRun();

  std::unique_ptr<G4SimEvent> produce(const edm::Event& inpevt,
                                      const edm::EventSetup& es,
                                      RunManagerMT& runManagerMaster);

  void abortEvent();
  void abortRun(bool softAbort = false);

  inline G4SimEvent* simEvent() { return m_simEvent; }

  void Connect(RunAction*);
  void Connect(EventAction*);
  void Connect(TrackingAction*);
  void Connect(SteppingAction*);

  SimTrackManager* GetSimTrackManager();
  std::vector<SensitiveTkDetector*>& sensTkDetectors();
  std::vector<SensitiveCaloDetector*>& sensCaloDetectors();
  std::vector<SimWatcher*>& producers();

  void initializeG4(RunManagerMT* runManagerMaster, const edm::EventSetup& es);

private:
  void initializeUserActions();

  void initializeRun();
  void terminateRun();

  G4Event* generateEvent(const edm::Event& inpevt);
  void resetGenParticleId(const edm::Event& inpevt);

  void DumpMagneticField(const G4Field*, const std::string&) const;

  Generator m_generator;
  edm::EDGetTokenT<edm::HepMCProduct> m_InToken;
  edm::EDGetTokenT<edm::HepMCProduct> m_LHCToken;
  edm::EDGetTokenT<edm::LHCTransportLinkContainer> m_theLHCTlinkToken;
  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> m_MagField;
  const MagneticField* m_pMagField{nullptr};

  bool m_nonBeam{false};
  bool m_pUseMagneticField{true};
  bool m_hasWatchers{false};
  bool m_LHCTransport{false};
  bool m_threadInitialized{false};
  bool m_runTerminated{false};
  bool m_dumpMF{false};
  int m_EvtMgrVerbosity{0};

  const int m_thread_index{-1};
  edm::RunNumber_t m_currentRunNumber{0};

  edm::ParameterSet m_pField;
  edm::ParameterSet m_pRunAction;
  edm::ParameterSet m_pEventAction;
  edm::ParameterSet m_pStackingAction;
  edm::ParameterSet m_pTrackingAction;
  edm::ParameterSet m_pSteppingAction;
  edm::ParameterSet m_pCustomUIsession;
  edm::ParameterSet m_p;

  std::unique_ptr<G4RunManagerKernel> m_kernel;
  std::unique_ptr<RunAction> m_userRunAction;
  std::unique_ptr<SimRunInterface> m_runInterface;
  std::unique_ptr<SimActivityRegistry> m_registry;
  std::unique_ptr<SimTrackManager> m_trackManager;
  std::unique_ptr<G4Event> m_currentEvent;
  std::unique_ptr<CMSSteppingVerbose> m_sVerbose{nullptr};

  G4Run* m_currentRun{nullptr};
  G4SimEvent* m_simEvent{nullptr};

  std::vector<SensitiveTkDetector*> m_sensTkDets;
  std::vector<SensitiveCaloDetector*> m_sensCaloDets;
  std::vector<SimWatcher*> m_watchers;

  std::unordered_map<std::string, std::unique_ptr<SensitiveDetectorMakerBase>> m_sdMakers;
};

#endif
