#ifndef SimG4Core_RunManager_H
#define SimG4Core_RunManager_H

#include "FWCore/EDProduct/interface/EDProduct.h"
#include "FWCore/CoreFramework/interface/Event.h"
#include "FWCore/CoreFramework/interface/Handle.h"
#include "FWCore/CoreFramework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SealKernel/Context.h"

class PrimaryTransformer;
class Generator;
class Physics;
class G4SimEvent;

class G4RunManagerKernel;
class G4Run;
class G4Event;
class G4UserRunAction;
 
class RunManager
{
public:
    static RunManager * instance();
    static RunManager * init(edm::ParameterSet const & p); 
    virtual ~RunManager();
    void initializeUserActions();
    void initializeRun();
    void terminateRun();
    void abortRun(bool softAbort=false);
    const G4Run * currentRun() const { return m_currentRun; }
    void produce(const edm::EventSetup & es);
    void abortEvent();
    const Generator * generator() const { return m_generator; }
    const G4Event * currentEvent() const { return m_currentEvent; }
    G4SimEvent * simEvent() { return m_simEvent; }
    void maybeInitializeManager(const edm::EventSetup &);
protected:
    G4Event * generateEvent(int evt);
private:
    static RunManager * me;
    explicit RunManager(edm::ParameterSet const & p);
    seal::Handle<seal::Context> m_context;
    edm::ParameterSet m_paramSet;
    G4RunManagerKernel * m_kernel;
    Generator * m_generator;
    Physics * m_physics;
    PrimaryTransformer * m_primaryTransformer;
    bool m_managerInitialized;
    bool m_geometryInitialized;
    bool m_physicsInitialized;
    bool m_runInitialized;
    bool m_runTerminated;
    bool m_runAborted;
    G4Run * m_currentRun;
    G4Event * m_currentEvent;
    G4SimEvent * m_simEvent;
    G4UserRunAction * m_userRunAction;
};

#endif
