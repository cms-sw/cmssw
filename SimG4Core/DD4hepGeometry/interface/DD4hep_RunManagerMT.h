#ifndef SimG4Core_DD4hep_RunManagerMT_H
#define SimG4Core_DD4hep_RunManagerMT_H

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

namespace cms {
  class DDDetector;
  struct DDSpecParRegistry;
}

class DDDWorld;

class DD4hep_DDG4ProductionCuts;
class MagneticField;

class G4MTRunManagerKernel;
class G4Run;
class G4Event;
class G4Field;
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

class DD4hep_RunManagerMT 
{
  friend class RunManagerMTWorker;

public:
  explicit DD4hep_RunManagerMT(edm::ParameterSet const &);
  ~DD4hep_RunManagerMT();

  void initG4(const cms::DDDetector*, const cms::DDSpecParRegistry*, const MagneticField*, const HepPDT::ParticleDataTable*);

  void initializeUserActions();

  void stopG4();

  void Connect(RunAction*);

  // Keep this to keep ExceptionHandler to compile, probably removed
  // later (or functionality moved to RunManagerMTWorker)
  inline void abortRun(bool softAbort=false) {}

  inline const DDDWorld& world() const {
    return *m_world;
  }

  inline const SensitiveDetectorCatalog& catalog() const {
    return m_catalog;
  }

  inline const std::vector<std::string>& G4Commands() const {
    return m_G4Commands;
  }

  // In order to share the physics list with the worker threads, we
  // need a non-const pointer. Thread-safety is handled inside Geant4
  // with TLS. 
  inline PhysicsList *physicsListForWorker() const {
    return m_physicsList.get();
  }

private:
  void terminateRun();
  void DumpMagneticField(const G4Field*) const;

  G4MTRunManagerKernel* m_kernel;
    
  std::unique_ptr<CustomUIsession> m_UIsession;
  std::unique_ptr<PhysicsList> m_physicsList;
  bool m_managerInitialized;
  bool m_runTerminated;
  bool m_pUseMagneticField;
  RunAction* m_userRunAction;
  G4Run* m_currentRun;
  G4StateManager* m_stateManager;
  G4GeometryManager* m_geometryManager;

  std::unique_ptr<SimRunInterface> m_runInterface;

  const std::string m_PhysicsTablesDir;
  bool m_StorePhysicsTables;
  bool m_RestorePhysicsTables;
  bool m_check;
  edm::ParameterSet m_pField;
  edm::ParameterSet m_pPhysics; 
  edm::ParameterSet m_pRunAction;      
  edm::ParameterSet m_g4overlap;
  std::vector<std::string> m_G4Commands;
  edm::ParameterSet m_p;

  std::unique_ptr<DDDWorld> m_world;
  std::unique_ptr<DD4hep_DDG4ProductionCuts> m_prodCuts;
  SimActivityRegistry m_registry;
  SensitiveDetectorCatalog m_catalog;
    
  std::string m_FieldFile;
  std::string m_WriteFile;
  std::string m_RegionFile;
};

#endif
