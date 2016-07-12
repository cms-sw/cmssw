#include "SimG4Core/Application/interface/SimRunInterface.h"
#include "SimG4Core/Application/interface/RunManager.h"
#include "SimG4Core/Application/interface/RunManagerMT.h"
#include "SimG4Core/Application/interface/RunManagerMTWorker.h"
#include "SimG4Core/Application/interface/RunAction.h"
#include "SimG4Core/Application/interface/EventAction.h"
#include "SimG4Core/Application/interface/TrackingAction.h"
#include "SimG4Core/Application/interface/SteppingAction.h"
#include "SimG4Core/Application/interface/G4SimEvent.h"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

SimRunInterface::SimRunInterface(RunManager* runm, bool master)
  : m_runManager(runm),m_runManagerMT(nullptr),m_runManagerMTWorker(nullptr),
    m_SimTrackManager(nullptr),
    m_isMaster(master)
{
  if(m_runManager) {
    m_SimTrackManager = m_runManager->GetSimTrackManager();
  } 
}

SimRunInterface::SimRunInterface(RunManagerMT* runm, bool master)
  : m_runManager(nullptr),m_runManagerMT(runm),m_runManagerMTWorker(nullptr),
    m_SimTrackManager(nullptr),m_isMaster(master)
{}

SimRunInterface::SimRunInterface(RunManagerMTWorker* runm, bool master)
  : m_runManager(nullptr),m_runManagerMT(nullptr),m_runManagerMTWorker(runm),
    m_SimTrackManager(nullptr),m_isMaster(master)
{
  if(m_runManagerMTWorker) {
    m_SimTrackManager = m_runManagerMTWorker->GetSimTrackManager();
  }
}

SimRunInterface::~SimRunInterface()
{}

void SimRunInterface::setRunManagerMTWorker(RunManagerMTWorker *run) {
  m_runManagerMTWorker = run;
}

void SimRunInterface::Connect(RunAction* runAction)
{
  if(m_runManager) {
    m_runManager->Connect(runAction);
  } else if(m_runManagerMT) {
    m_runManagerMT->Connect(runAction);
  } else if(m_runManagerMTWorker) {
    m_runManagerMTWorker->Connect(runAction);
  }
}

void SimRunInterface::Connect(EventAction* eventAction)
{
  if(m_runManager) {
    m_runManager->Connect(eventAction);
  } else if(m_runManagerMTWorker) {
    m_runManagerMTWorker->Connect(eventAction);
  }
}

void SimRunInterface::Connect(TrackingAction* trackAction)
{
  if(m_runManager) {
    m_runManager->Connect(trackAction);
  } else if(m_runManagerMTWorker) {
    m_runManagerMTWorker->Connect(trackAction);
  }
}

void SimRunInterface::Connect(SteppingAction* stepAction)
{
  if(m_runManager) {
    m_runManager->Connect(stepAction);
  } else if(m_runManagerMTWorker) {
    m_runManagerMTWorker->Connect(stepAction);
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
  } else if(m_runManagerMTWorker) {
    m_runManagerMTWorker->abortEvent();
  }  
}

void SimRunInterface::abortRun(bool softAbort)
{
  if(m_runManager) {
    m_runManager->abortRun(softAbort);
  } else if(m_runManagerMTWorker) {
    m_runManagerMTWorker->abortRun(softAbort);
  }  
}

G4SimEvent* SimRunInterface::simEvent()
{
  G4SimEvent* ptr = 0;
  if(m_runManager) {
    ptr = m_runManager->simEvent();
  } else if(m_runManagerMTWorker) {
    ptr = m_runManagerMTWorker->simEvent();
  }  
  return ptr;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
