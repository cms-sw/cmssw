#ifndef SimG4Core_RunManagerMT_H
#define SimG4Core_RunManagerMT_H

#include <memory>
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "SimG4Core/Geometry/interface/SensitiveDetectorCatalog.h"

#include "SimG4Core/Notification/interface/SimActivityRegistry.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include <memory>
#include "boost/shared_ptr.hpp"

namespace CLHEP {
  class HepJamesRandom;
}

namespace sim {
  class FieldBuilder;
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

class G4RunManagerKernel;
class G4Run;
class G4Event;
class G4Field;
class RunAction;

class SimRunInterface;
//class ExceptionHandler;

namespace HepPDT {
  class ParticleDataTable;
}

/**
 * RunManagerMT should be constructed in a newly spanned thread
 * (acting as the Geant4 master thread), and there should be exactly
 * one instance of it. 
 */

class RunManagerMT 
{
public:
  RunManagerMT(edm::ParameterSet const & p);
  ~RunManagerMT();

  void initG4(const DDCompactView *pDD, const MagneticField *pMF, const HepPDT::ParticleDataTable *fPDGTable, const edm::EventSetup & es);

  void initializeUserActions();

  void stopG4();

  void             Connect(RunAction*);

  // Keep this to keep ExceptionHandler to compile, probably removed
  // later (or functionality moved to RunManagerMTWorker)
  void abortRun(bool softAbort=false) {}


  const DDDWorld& world() const {
    return *m_world;
  }

  const SensitiveDetectorCatalog& catalog() const {
    return m_catalog;
  }

  // In order to share the physics list with the worker threads, we
  // need a non-const pointer. Thread-safety is handled inside Geant4
  // with TLS. Should we consider a friend declaration here in order
  // to avoid misuse?
  PhysicsList *physicsListForWorker() const {
    return m_physicsList.get();
  }

private:
  void terminateRun();
  void DumpMagneticField( const G4Field*) const;

  G4RunManagerKernel * m_kernel;
    
  Generator * m_generator;
  std::string m_InTag ;
    
  std::unique_ptr<PhysicsList> m_physicsList;
  bool m_managerInitialized;
  bool m_runTerminated;
  bool m_runAborted;
  bool firstRun;
  const bool m_pUseMagneticField;
  std::unique_ptr<RunAction> m_userRunAction;
  std::unique_ptr<SimRunInterface> m_runInterface;

  //edm::EDGetTokenT<edm::HepMCProduct> m_HepMC;

  std::string m_PhysicsTablesDir;
  bool m_StorePhysicsTables;
  bool m_RestorePhysicsTables;
  bool m_check;
  edm::ParameterSet m_pGeometry;
  edm::ParameterSet m_pField;
  edm::ParameterSet m_pGenerator;   
  edm::ParameterSet m_pVertexGenerator;
  edm::ParameterSet m_pPhysics; 
  edm::ParameterSet m_pRunAction;      
  std::vector<std::string> m_G4Commands;
  edm::ParameterSet m_p;
  //ExceptionHandler* m_CustomExceptionHandler ;

  std::unique_ptr<DDDWorld> m_world;
  SimActivityRegistry m_registry;
  SensitiveDetectorCatalog m_catalog;
    
  sim::FieldBuilder             *m_fieldBuilder;
    
  edm::InputTag m_theLHCTlinkTag;

  std::string m_FieldFile;
  std::string m_WriteFile;
};

#endif
