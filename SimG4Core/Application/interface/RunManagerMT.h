#ifndef SimG4Core_RunManagerMT_H
#define SimG4Core_RunManagerMT_H

#include <memory>
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "SimG4Core/SensitiveDetector/interface/AttachSD.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveDetector.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveTkDetector.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveCaloDetector.h"

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
class SimTrackManager;

class RunAction;
class EventAction; 
class TrackingAction;
class SteppingAction;

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

  void stopG4();

  // Keep these to keep SimTrackManager compiling for now, probably to
  // be moved to RunManagerMTWorker
  void abortRun(bool softAbort=false) {}
  void abortEvent() {}
  G4SimEvent * simEvent() { return nullptr; }
  SimTrackManager* GetSimTrackManager() { return nullptr; }
  void             Connect(RunAction*) {}
  void             Connect(EventAction*) {}
  void             Connect(TrackingAction*) {}
  void             Connect(SteppingAction*) {}


  const DDDWorld& world() const {
    return *m_world;
  }

  // In order to share the physics list with the worker threads, we
  // need a non-const pointer. Thread-safety is handled inside Geant4
  // with TLS. Should we consider a friend declaration here in order
  // to avoid misuse?
  PhysicsList *physicsListForWorker() const {
    return m_physicsList.get();
  }

protected:


private:

  G4RunManagerKernel * m_kernel;
    
  Generator * m_generator;
  std::string m_InTag ;
    
  bool m_nonBeam;
  std::unique_ptr<PhysicsList> m_physicsList;
  PrimaryTransformer * m_primaryTransformer;
  bool m_managerInitialized;
  bool m_runInitialized;
  bool m_runTerminated;
  bool m_runAborted;
  bool firstRun;
  bool m_pUseMagneticField;

  //edm::EDGetTokenT<edm::HepMCProduct> m_HepMC;

  std::string m_PhysicsTablesDir;
  bool m_StorePhysicsTables;
  bool m_RestorePhysicsTables;
  int m_EvtMgrVerbosity;
  bool m_check;
  edm::ParameterSet m_pGeometry;
  edm::ParameterSet m_pField;
  edm::ParameterSet m_pGenerator;   
  edm::ParameterSet m_pVertexGenerator;
  edm::ParameterSet m_pPhysics; 
  edm::ParameterSet m_pRunAction;      
  edm::ParameterSet m_pEventAction;
  edm::ParameterSet m_pStackingAction;
  edm::ParameterSet m_pTrackingAction;
  edm::ParameterSet m_pSteppingAction;
  std::vector<std::string> m_G4Commands;
  edm::ParameterSet m_p;
  //ExceptionHandler* m_CustomExceptionHandler ;

  std::unique_ptr<DDDWorld> m_world;
  AttachSD * m_attach;
  std::vector<SensitiveTkDetector*> m_sensTkDets;
  std::vector<SensitiveCaloDetector*> m_sensCaloDets;

  SimActivityRegistry m_registry;
  std::vector<boost::shared_ptr<SimWatcher> > m_watchers;
  std::vector<boost::shared_ptr<SimProducer> > m_producers;
    
  std::auto_ptr<SimTrackManager> m_trackManager;
  sim::FieldBuilder             *m_fieldBuilder;
    
  edm::InputTag m_theLHCTlinkTag;

  std::string m_FieldFile;
  std::string m_WriteFile;
};

#endif
