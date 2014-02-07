#include "SimG4Core/Generators/interface/Generator.h"
#include "SimG4Core/Application/interface/RunManager.h"
#include "SimG4Core/Application/interface/SimRunInterface.h"
#include "SimG4Core/Application/interface/ActionInitialization.h"
#include "SimG4Core/Application/interface/RunAction.h"
#include "SimG4Core/Application/interface/EventAction.h"
#include "SimG4Core/Application/interface/TrackingAction.h"
#include "SimG4Core/Application/interface/SteppingAction.h"
#include "SimG4Core/Application/interface/StackingAction.h"
#include "SimG4Core/Application/interface/SimTrackManager.h"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

ActionInitialization::ActionInitialization(const edm::ParameterSet & p, 
					   RunManager* runm)
  : G4VUserActionInitialization(),
    m_runManager(runm),
    m_pGenerator(p.getParameter<edm::ParameterSet>("Generator")),
    m_pPhysics(p.getParameter<edm::ParameterSet>("Physics")),
    m_pRunAction(p.getParameter<edm::ParameterSet>("RunAction")),      
    m_pEventAction(p.getParameter<edm::ParameterSet>("EventAction")),
    m_pStackingAction(p.getParameter<edm::ParameterSet>("StackingAction")),
    m_pTrackingAction(p.getParameter<edm::ParameterSet>("TrackingAction")),
    m_pSteppingAction(p.getParameter<edm::ParameterSet>("SteppingAction"))
{}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

ActionInitialization::~ActionInitialization()
{}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void ActionInitialization::BuildForMaster() const
{
  SimRunInterface* interface = new SimRunInterface(m_runManager, true);

  RunAction* runAction = new RunAction(m_pRunAction, interface);
  SetUserAction(runAction);
  interface->Connect(runAction);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#include "G4AutoLock.hh"
namespace { G4Mutex ActionInitializationMutex = G4MUTEX_INITIALIZER; }

void ActionInitialization::Build() const
{
  G4AutoLock l(&ActionInitializationMutex);

  SimRunInterface* interface = new SimRunInterface(m_runManager, false);

  //Generator* gen = new Generator(m_pGenerator);
  //SetUserAction(gen);

  RunAction* runAction = new RunAction(m_pRunAction, interface);
  SetUserAction(runAction);
  interface->Connect(runAction);

  EventAction* eventAction = new EventAction(m_pEventAction, interface, 
					     interface->GetSimTrackManager());
  SetUserAction(eventAction);
  interface->Connect(eventAction);

  TrackingAction* trackAction = 
    new TrackingAction(eventAction, m_pTrackingAction);
  SetUserAction(trackAction);
  interface->Connect(trackAction);
  
  SteppingAction* stepAction = 
    new SteppingAction(eventAction, m_pSteppingAction);
  SetUserAction(stepAction);
  interface->Connect(stepAction);

  SetUserAction(new StackingAction(m_pStackingAction));
}  

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
