#ifndef SimG4Core_Application_RunManagerMTWorker_H
#define SimG4Core_Application_RunManagerMTWorker_H

#include "FWCore/Utilities/interface/EDGetToken.h"
#include "DataFormats/Provenance/interface/RunID.h"

#include "SimG4Core/Generators/interface/Generator.h"
#include "SimDataFormats/Forward/interface/LHCTransportLinkContainer.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "SimG4Core/Notification/interface/G4SimEvent.h"

#include <memory>
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

class G4Event;
class G4Run;
class SimTrackManager;
class CustomUIsession;

class RunAction;
class EventAction;
class TrackingAction;
class SteppingAction;
class CMSSteppingVerbose;
class CMSSimEventManager;
class G4Field;

class SensitiveTkDetector;
class SensitiveCaloDetector;
class SensitiveDetectorMakerBase;

class SimWatcher;
class SimProducer;

class RunManagerMTWorker {
public:
  explicit RunManagerMTWorker(const edm::ParameterSet& iConfig, edm::ConsumesCollector&& iC);
  ~RunManagerMTWorker();

  void beginRun(const edm::EventSetup&);
  void endRun();

  G4SimEvent* produce(const edm::Event& inpevt, const edm::EventSetup& es, RunManagerMT& runManagerMaster);

  void abortEvent();
  void abortRun(bool softAbort = false);

  void Connect(RunAction*);
  void Connect(EventAction*);
  void Connect(TrackingAction*);
  void Connect(SteppingAction*);

  SimTrackManager* GetSimTrackManager();
  std::vector<SensitiveTkDetector*>& sensTkDetectors();
  std::vector<SensitiveCaloDetector*>& sensCaloDetectors();
  std::vector<std::shared_ptr<SimProducer>>& producers();

  void initializeG4(RunManagerMT* runManagerMaster, const edm::EventSetup& es);

  inline G4SimEvent* simEvent() { return &m_simEvent; }
  inline int getThreadIndex() const { return m_thread_index; }

private:
  void initializeTLS();
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
  bool m_UseG4EventManager{true};
  bool m_pUseMagneticField{true};
  bool m_hasWatchers{false};
  bool m_LHCTransport{false};
  bool m_dumpMF{false};
  bool m_endOfRun{false};

  const int m_thread_index{-1};

  edm::ParameterSet m_pField;
  edm::ParameterSet m_pRunAction;
  edm::ParameterSet m_pEventAction;
  edm::ParameterSet m_pStackingAction;
  edm::ParameterSet m_pTrackingAction;
  edm::ParameterSet m_pSteppingAction;
  edm::ParameterSet m_pCustomUIsession;
  std::vector<std::string> m_G4CommandsEndRun;
  edm::ParameterSet m_p;

  struct TLSData;
  TLSData* m_tls{nullptr};

  CustomUIsession* m_UIsession{nullptr};
  G4SimEvent m_simEvent;
  std::unique_ptr<CMSSimEventManager> m_evtManager;
  std::unique_ptr<CMSSteppingVerbose> m_sVerbose;
  std::unordered_map<std::string, std::unique_ptr<SensitiveDetectorMakerBase>> m_sdMakers;
};

#endif
