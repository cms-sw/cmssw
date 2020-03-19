#ifndef SimG4Core_RunManagerMT_H
#define SimG4Core_RunManagerMT_H

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "SimG4Core/Geometry/interface/SensitiveDetectorCatalog.h"

#include "SimG4Core/Notification/interface/SimActivityRegistry.h"

#include <memory>

class PrimaryTransformer;
class Generator;
class PhysicsList;
class CustomUIsession;

class SimWatcher;
class SimProducer;
class G4SimEvent;

class RunAction;

class DDCompactView;

namespace cms {
  class DDCompactView;
}

class DDDWorld;

class G4MTRunManagerKernel;
class G4Run;
class G4Event;
class G4StateManager;
class G4GeometryManager;
class RunAction;

class SimRunInterface;

namespace HepPDT {
  class ParticleDataTable;
}

/**
 * RunManagerMT should be constructed in a newly spanned thread
 * (acting as the Geant4 master thread), and there should be exactly
 * one instance of it. 
 */
class RunManagerMTWorker;

class RunManagerMT {
  friend class RunManagerMTWorker;

public:
  explicit RunManagerMT(edm::ParameterSet const&);
  ~RunManagerMT();

  //  void initG4(const DDCompactView*, const cms::DDCompactView*, const MagneticField*, const HepPDT::ParticleDataTable*);
  void initG4(const DDCompactView*, const cms::DDCompactView*, const HepPDT::ParticleDataTable*);

  void initializeUserActions();

  void stopG4();

  void Connect(RunAction*);

  // Keep this to keep ExceptionHandler to compile, probably removed
  // later (or functionality moved to RunManagerMTWorker)
  inline void abortRun(bool softAbort = false) {}

  inline const DDDWorld& world() const { return *m_world; }

  inline const SensitiveDetectorCatalog& catalog() const { return m_catalog; }

  inline const std::vector<std::string>& G4Commands() const { return m_G4Commands; }

  // In order to share the physics list with the worker threads, we
  // need a non-const pointer. Thread-safety is handled inside Geant4
  // with TLS.
  inline PhysicsList* physicsListForWorker() const { return m_physicsList.get(); }

private:
  void terminateRun();

  G4MTRunManagerKernel* m_kernel;

  CustomUIsession* m_UIsession;
  std::unique_ptr<PhysicsList> m_physicsList;
  bool m_managerInitialized;
  bool m_runTerminated;
  RunAction* m_userRunAction;
  G4Run* m_currentRun;
  G4StateManager* m_stateManager;
  G4GeometryManager* m_geometryManager;

  std::unique_ptr<SimRunInterface> m_runInterface;

  const std::string m_PhysicsTablesDir;
  bool m_StorePhysicsTables;
  bool m_RestorePhysicsTables;
  bool m_UseParametrisedEMPhysics;
  bool m_check;
  edm::ParameterSet m_pPhysics;
  edm::ParameterSet m_pRunAction;
  edm::ParameterSet m_g4overlap;
  std::vector<std::string> m_G4Commands;
  edm::ParameterSet m_p;

  std::unique_ptr<DDDWorld> m_world;
  SimActivityRegistry m_registry;
  SensitiveDetectorCatalog m_catalog;
};

#endif
