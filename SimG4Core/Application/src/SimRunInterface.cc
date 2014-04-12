#include "SimG4Core/Application/interface/SimRunInterface.h"
#include "SimG4Core/Application/interface/RunManager.h"
#include "SimG4Core/Application/interface/RunAction.h"
#include "SimG4Core/Application/interface/EventAction.h"
#include "SimG4Core/Application/interface/TrackingAction.h"
#include "SimG4Core/Application/interface/SteppingAction.h"
#include "SimG4Core/Application/interface/G4SimEvent.h"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

SimRunInterface::SimRunInterface(RunManager* runm, bool master)
  : m_runManager(runm),m_SimTrackManager(0),m_isMaster(master)
{
  if(m_runManager) {
    m_SimTrackManager = m_runManager->GetSimTrackManager();
  }
}

SimRunInterface::~SimRunInterface()
{}

void SimRunInterface::Connect(RunAction* runAction)
{
  if(m_runManager) {
    m_runManager->Connect(runAction);
  }
}

void SimRunInterface::Connect(EventAction* eventAction)
{
  if(m_runManager) {
    m_runManager->Connect(eventAction);
  }
}

void SimRunInterface::Connect(TrackingAction* trackAction)
{
  if(m_runManager) {
    m_runManager->Connect(trackAction);
  }
}

void SimRunInterface::Connect(SteppingAction* stepAction)
{
  if(m_runManager) {
    m_runManager->Connect(stepAction);
  }
}

SimTrackManager* SimRunInterface::GetSimTrackManager()
{
  return m_SimTrackManager;
}

void SimRunInterface::abortEvent()
{
  if(m_runManager) {
    m_runManager->abortEvent();
  }  
}

void SimRunInterface::abortRun(bool softAbort)
{
  if(m_runManager) {
    m_runManager->abortRun(softAbort);
  }  
}

G4SimEvent* SimRunInterface::simEvent()
{
  G4SimEvent* ptr = 0;
  if(m_runManager) {
    ptr = m_runManager->simEvent();
  }  
  return ptr;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
