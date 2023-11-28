#ifndef SimG4Core_RunManagerMT_H
#define SimG4Core_RunManagerMT_H

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

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

  void initG4(const DDCompactView*, const cms::DDCompactView*, const HepPDT::ParticleDataTable*);

  void initializeUserActions();

  void stopG4();

  void Connect(RunAction*);

  // Keep this to keep ExceptionHandler to compile, probably removed
  // later (or functionality moved to RunManagerMTWorker).
  //  inline void abortRun(bool softAbort = false) {}

  inline const DDDWorld& world() const { return *m_world; }

  inline const SensitiveDetectorCatalog& catalog() const { return m_catalog; }

  inline const std::vector<std::string>& G4Commands() const { return m_G4Commands; }

  // In order to share the physics list with the worker threads, we
  // need a non-const pointer. Thread-safety is handled inside Geant4.
  inline PhysicsList* physicsListForWorker() const { return m_physicsList.get(); }

  inline bool isPhase2() const { return m_isPhase2; }

private:
  void terminateRun();

  void checkVoxels();

  void setupVoxels();

  void runForPhase2();

  G4MTRunManagerKernel* m_kernel;

  CustomUIsession* m_UIsession;
  std::unique_ptr<PhysicsList> m_physicsList;
  bool m_managerInitialized{false};
  bool m_runTerminated{false};
  RunAction* m_userRunAction{nullptr};
  G4Run* m_currentRun{nullptr};
  G4StateManager* m_stateManager;

  std::unique_ptr<SimRunInterface> m_runInterface;

  const std::string m_PhysicsTablesDir;
  bool m_StorePhysicsTables;
  bool m_RestorePhysicsTables;
  bool m_check;
  bool m_geoFromDD4hep;
  bool m_score;
  bool m_isPhase2{false};
  int m_stepverb;
  std::string m_regionFile{""};
  edm::ParameterSet m_pPhysics;
  edm::ParameterSet m_pRunAction;
  edm::ParameterSet m_CheckOverlap;
  edm::ParameterSet m_Init;
  std::vector<std::string> m_G4Commands;

  std::unique_ptr<DDDWorld> m_world;
  SimActivityRegistry m_registry;
  SensitiveDetectorCatalog m_catalog;
};

#endif
