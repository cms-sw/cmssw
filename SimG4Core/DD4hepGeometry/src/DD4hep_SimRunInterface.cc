#include "SimG4Core/DD4hepGeometry/interface/DD4hep_SimRunInterface.h"
#include "SimG4Core/DD4hepGeometry/interface/DD4hep_RunManagerMT.h"
#include "SimG4Core/DD4hepGeometry/interface/DD4hep_RunManagerMTWorker.h"
#include "SimG4Core/Application/interface/RunAction.h"
#include "SimG4Core/Application/interface/EventAction.h"
#include "SimG4Core/Application/interface/TrackingAction.h"
#include "SimG4Core/Application/interface/SteppingAction.h"
#include "SimG4Core/Notification/interface/G4SimEvent.h"

SimRunInterface::SimRunInterface(DD4hep_RunManagerMT* runm, bool master)
  : m_runManagerMT(runm),m_runManagerMTWorker(nullptr),
    m_SimTrackManager(nullptr),m_isMaster(master)
{}

SimRunInterface::SimRunInterface(DD4hep_RunManagerMTWorker* runm, bool master)
  : m_runManagerMT(nullptr),m_runManagerMTWorker(runm),
    m_SimTrackManager(nullptr),m_isMaster(master)
{
  if(m_runManagerMTWorker) {
    m_SimTrackManager = m_runManagerMTWorker->GetSimTrackManager();
  }
}

SimRunInterface::~SimRunInterface()
{}

void
SimRunInterface::setRunManagerMTWorker(DD4hep_RunManagerMTWorker *run) {
  m_runManagerMTWorker = run;
}

void
SimRunInterface::Connect(RunAction* runAction)
{
  if(m_runManagerMT) {
    m_runManagerMT->Connect(runAction);
  } else if(m_runManagerMTWorker) {
    m_runManagerMTWorker->Connect(runAction);
  }
}

void
SimRunInterface::Connect(EventAction* eventAction)
{
  if(m_runManagerMTWorker) {
    m_runManagerMTWorker->Connect(eventAction);
  }
}

void
SimRunInterface::Connect(TrackingAction* trackAction)
{
  if(m_runManagerMTWorker) {
    m_runManagerMTWorker->Connect(trackAction);
  }
}

void
SimRunInterface::Connect(SteppingAction* stepAction)
{
  if(m_runManagerMTWorker) {
    m_runManagerMTWorker->Connect(stepAction);
  }
}

SimTrackManager* SimRunInterface::GetSimTrackManager()
{
  return m_SimTrackManager;
}

void
SimRunInterface::abortEvent()
{
  if(m_runManagerMTWorker) {
    m_runManagerMTWorker->abortEvent();
  }  
}

void
SimRunInterface::abortRun(bool softAbort)
{
  if(m_runManagerMTWorker) {
    m_runManagerMTWorker->abortRun(softAbort);
  }  
}

G4SimEvent*
SimRunInterface::simEvent()
{
  G4SimEvent* ptr = nullptr;
  if(m_runManagerMTWorker) {
    ptr = m_runManagerMTWorker->simEvent();
  }  
  return ptr;
}
