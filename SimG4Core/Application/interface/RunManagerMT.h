#ifndef SimG4Core_RunManagerMT_H
#define SimG4Core_RunManagerMT_H

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "SimG4Core/Geometry/interface/SensitiveDetectorCatalog.h"

#include "SimG4Core/Notification/interface/SimActivityRegistry.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include <memory>

namespace CLHEP {
  class HepJamesRandom;
}

namespace sim {
  class FieldBuilder;
  class ChordFinderSetter;
}

class PrimaryTransformer;
class Generator;
class PhysicsList;

class SimWatcher;
class SimProducer;
class G4SimEvent;

class RunAction;

class DDCompactView;
class DDDWorld;
class MagneticField;

class G4MTRunManagerKernel;
class G4Run;
class G4Event;
class G4Field;
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

class RunManagerMT 
{
  friend class RunManagerMTWorker;

public:
  RunManagerMT(edm::ParameterSet const & p);
  ~RunManagerMT();

  void initG4(const DDCompactView *pDD, const MagneticField *pMF, const HepPDT::ParticleDataTable *fPDGTable);

  void initializeUserActions();

  void stopG4();

  void Connect(RunAction*);

  // Keep this to keep ExceptionHandler to compile, probably removed
  // later (or functionality moved to RunManagerMTWorker)
  void abortRun(bool softAbort=false) {}

  const DDDWorld& world() const {
    return *m_world;
  }

  const SensitiveDetectorCatalog& catalog() const {
    return m_catalog;
  }

  const std::vector<std::string>& G4Commands() const {
    return m_G4Commands;
  }

  // In order to share the physics list with the worker threads, we
  // need a non-const pointer. Thread-safety is handled inside Geant4
  // with TLS. Should we consider a friend declaration here in order
  // to avoid misuse?
  PhysicsList *physicsListForWorker() const {
    return m_physicsList.get();
  }

  // In order to share the ChordFinderSetter (for
  // G4MonopoleTransportation) with the worker threads, we need a
  // non-const pointer. Thread-safety is handled inside
  // ChordFinderStter with TLS. Should we consider a friend
  // declaration here in order to avoid misuse?
  sim::ChordFinderSetter *chordFinderSetterForWorker() const {
    return m_chordFinderSetter.get();
  }

private:
  void terminateRun();
  void DumpMagneticField( const G4Field*) const;

  G4MTRunManagerKernel * m_kernel;
    
  std::unique_ptr<PhysicsList> m_physicsList;
  bool m_managerInitialized;
  bool m_runTerminated;
  bool m_pUseMagneticField;
  RunAction* m_userRunAction;
  G4Run* m_currentRun;
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

  std::unique_ptr<DDDWorld> m_world;
  SimActivityRegistry m_registry;
  SensitiveDetectorCatalog m_catalog;
    
  sim::FieldBuilder             *m_fieldBuilder;
  std::unique_ptr<sim::ChordFinderSetter> m_chordFinderSetter;
    
  std::string m_FieldFile;
  std::string m_WriteFile;
  std::string m_RegionFile;
};

#endif
