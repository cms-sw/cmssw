#ifndef SimG4Core_RunManager_H
#define SimG4Core_RunManager_H

#include <memory>
#include "FWCore/Framework/interface/Event.h"
// #include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "SimG4Core/SensitiveDetector/interface/AttachSD.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveDetector.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveTkDetector.h"
#include "SimG4Core/SensitiveDetector/interface/SensitiveCaloDetector.h"

#include "SimG4Core/Notification/interface/SimActivityRegistry.h"

#include <memory>
#include "boost/shared_ptr.hpp"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "FWCore/Framework/interface/ESWatcher.h"

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

class DDDWorld;

class G4RunManagerKernel;
class G4Run;
class G4Event;
class G4UserRunAction;

class ExceptionHandler ;

class RunManager
{
public:
    RunManager(edm::ParameterSet const & p);
    //static RunManager * instance();
    //static RunManager * init(edm::ParameterSet const & p); 
    virtual ~RunManager();
    void initG4(const edm::EventSetup & es);
    void initializeUserActions();
    void initializeRun();
    void terminateRun();
    void abortRun(bool softAbort=false);
    const G4Run * currentRun() const { return m_currentRun; }
    void produce(edm::Event& inpevt, const edm::EventSetup& es);
    void abortEvent();
    const Generator * generator() const { return m_generator; }
    const G4Event * currentEvent() const { return m_currentEvent; }
    G4SimEvent * simEvent() { return m_simEvent; }
    std::vector<SensitiveTkDetector*>& sensTkDetectors() { return m_sensTkDets; }
    std::vector<SensitiveCaloDetector*>& sensCaloDetectors() { return m_sensCaloDets; }

    std::vector<boost::shared_ptr<SimProducer> > producers() const {
       return m_producers;
    }
protected:
    G4Event * generateEvent( edm::Event& inpevt );

    void resetGenParticleId( edm::Event& inpevt ); 
private:

    // static RunManager * me;
    // explicit RunManager(edm::ParameterSet const & p);
    
    G4RunManagerKernel * m_kernel;
    
    Generator * m_generator;
    // edm::InputTag m_InTag ;
    std::string m_InTag ;
    
    bool m_nonBeam;
    std::auto_ptr<PhysicsList> m_physicsList;
    PrimaryTransformer * m_primaryTransformer;
    bool m_managerInitialized;
    //    bool m_geometryInitialized;
    //    bool m_physicsInitialized;
    bool m_runInitialized;
    bool m_runTerminated;
    bool m_runAborted;
    bool firstRun;
    bool m_pUseMagneticField;
    G4Run * m_currentRun;
    G4Event * m_currentEvent;
    G4SimEvent * m_simEvent;
    G4UserRunAction * m_userRunAction;
    std::string m_PhysicsTablesDir;
    bool m_StorePhysicsTables;
    bool m_RestorePhysicsTables;
    int m_EvtMgrVerbosity;
    bool m_Override;
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
    ExceptionHandler* m_CustomExceptionHandler ;

    AttachSD * m_attach;
    std::vector<SensitiveTkDetector*> m_sensTkDets;
    std::vector<SensitiveCaloDetector*> m_sensCaloDets;

    SimActivityRegistry m_registry;
    std::vector<boost::shared_ptr<SimWatcher> > m_watchers;
    std::vector<boost::shared_ptr<SimProducer> > m_producers;
    
    std::auto_ptr<SimTrackManager> m_trackManager;
    sim::FieldBuilder             *m_fieldBuilder;
    
    edm::ESWatcher<IdealGeometryRecord> idealGeomRcdWatcher_;
    edm::ESWatcher<IdealMagneticFieldRecord> idealMagRcdWatcher_;

    edm::InputTag m_theLHCTlinkTag;

    std::string m_WriteFile;
};

#endif
