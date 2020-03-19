#ifndef SimG4Core_RunManager_H
#define SimG4Core_RunManager_H

#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "SimDataFormats/Forward/interface/LHCTransportLinkContainer.h"

#include "SimG4Core/SensitiveDetector/interface/AttachSD.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveDetector.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveTkDetector.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveCaloDetector.h"
#include "SimG4Core/Notification/interface/SimActivityRegistry.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include <memory>

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
  class ConsumesCollector;
  class HepMCProduct;
}  // namespace edm

class PrimaryTransformer;
class Generator;
class PhysicsList;

class SimWatcher;
class SimProducer;
class G4SimEvent;
class SimTrackManager;

class RunAction;
class EventAction;
class TrackingAction;
class SteppingAction;
class CMSSteppingVerbose;

class DDDWorld;
class CustomUIsession;

class G4RunManagerKernel;
class G4Run;
class G4Event;
class G4Field;
class RunAction;

class SimRunInterface;

class RunManager {
public:
  RunManager(edm::ParameterSet const& p, edm::ConsumesCollector&& i);
  ~RunManager();
  void initG4(const edm::EventSetup& es);
  void initializeUserActions();
  void initializeRun();

  void stopG4();
  void terminateRun();
  void abortRun(bool softAbort = false);

  const G4Run* currentRun() const { return m_currentRun; }
  void produce(edm::Event& inpevt, const edm::EventSetup& es);
  void abortEvent();
  const Generator* generator() const { return m_generator; }
  const G4Event* currentEvent() const { return m_currentEvent; }
  G4SimEvent* simEvent() { return m_simEvent; }
  std::vector<SensitiveTkDetector*>& sensTkDetectors() { return m_sensTkDets; }
  std::vector<SensitiveCaloDetector*>& sensCaloDetectors() { return m_sensCaloDets; }
  std::vector<std::shared_ptr<SimProducer> > producers() const { return m_producers; }

  SimTrackManager* GetSimTrackManager();
  void Connect(RunAction*);
  void Connect(EventAction*);
  void Connect(TrackingAction*);
  void Connect(SteppingAction*);

protected:
  G4Event* generateEvent(edm::Event& inpevt);
  void resetGenParticleId(edm::Event& inpevt);
  void DumpMagneticField(const G4Field*) const;

private:
  G4RunManagerKernel* m_kernel;

  Generator* m_generator;

  edm::EDGetTokenT<edm::HepMCProduct> m_HepMC;
  edm::EDGetTokenT<edm::LHCTransportLinkContainer> m_LHCtr;

  bool m_nonBeam;
  CustomUIsession* m_UIsession;
  std::unique_ptr<PhysicsList> m_physicsList;
  PrimaryTransformer* m_primaryTransformer;

  bool m_managerInitialized;
  bool m_runInitialized;
  bool m_runTerminated;
  bool m_runAborted;
  bool firstRun;
  bool m_pUseMagneticField;
  bool m_hasWatchers;

  G4Run* m_currentRun;
  G4Event* m_currentEvent;
  G4SimEvent* m_simEvent;
  RunAction* m_userRunAction;
  SimRunInterface* m_runInterface;

  std::string m_PhysicsTablesDir;
  bool m_StorePhysicsTables;
  bool m_RestorePhysicsTables;
  bool m_UseParametrisedEMPhysics;
  int m_EvtMgrVerbosity;
  bool m_check;
  edm::ParameterSet m_pField;
  edm::ParameterSet m_pGenerator;
  edm::ParameterSet m_pPhysics;
  edm::ParameterSet m_pRunAction;
  edm::ParameterSet m_pEventAction;
  edm::ParameterSet m_pStackingAction;
  edm::ParameterSet m_pTrackingAction;
  edm::ParameterSet m_pSteppingAction;
  edm::ParameterSet m_g4overlap;
  std::vector<std::string> m_G4Commands;
  edm::ParameterSet m_p;

  std::vector<SensitiveTkDetector*> m_sensTkDets;
  std::vector<SensitiveCaloDetector*> m_sensCaloDets;

  std::unique_ptr<CMSSteppingVerbose> m_sVerbose;
  SimActivityRegistry m_registry;
  std::vector<std::shared_ptr<SimWatcher> > m_watchers;
  std::vector<std::shared_ptr<SimProducer> > m_producers;

  std::unique_ptr<SimTrackManager> m_trackManager;

  edm::ESWatcher<IdealGeometryRecord> idealGeomRcdWatcher_;
  edm::ESWatcher<IdealMagneticFieldRecord> idealMagRcdWatcher_;

  std::string m_FieldFile;
  std::string m_WriteFile;
  std::string m_RegionFile;
};

#endif
