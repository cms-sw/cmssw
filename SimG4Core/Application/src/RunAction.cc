#include "SimG4Core/Application/interface/RunAction.h"
#include "SimG4Core/Application/interface/SimRunInterface.h"

#include "SimG4Core/Notification/interface/BeginOfRun.h"
#include "SimG4Core/Notification/interface/EndOfRun.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4Timer.hh" 
#include <iostream>
#include <fstream>

RunAction::RunAction(const edm::ParameterSet& p, SimRunInterface* rm, bool master) 
  : m_runInterface(rm), 
    m_stopFile(p.getParameter<std::string>("StopFile")),
    m_timer(nullptr), m_isMaster(master)
{}

RunAction::~RunAction()
{}

void RunAction::BeginOfRunAction(const G4Run * aRun)
{
  if (std::ifstream(m_stopFile.c_str()))
    {
      edm::LogWarning("SimG4CoreApplication")
        << "RunAction::BeginOfRunAction: termination signal received";
      m_runInterface->abortRun(true);
    }
  BeginOfRun r(aRun);
  m_beginOfRunSignal(&r);
  /*
  if (m_isMaster) {
    m_timer = new G4Timer();
    m_timer->Start();
  }
  */
}

void RunAction::EndOfRunAction(const G4Run * aRun)
{
  if (isMaster) {
    edm::LogInfo("SimG4CoreApplication") 
      << "RunAction: total number of events "  << aRun->GetNumberOfEvent();
    if(m_timer) {
      m_timer->Stop();
      edm::LogInfo("SimG4CoreApplication") 
	<< "RunAction: Master thread time  "  << *m_timer;
      // std::cout << "\n" << "Master thread time:  "  << *m_timer << std::endl;
      delete m_timer;
    }
  }
  if (std::ifstream(m_stopFile.c_str()))
    {
      edm::LogWarning("SimG4CoreApplication")
        << "RunAction::EndOfRunAction: termination signal received";
      m_runInterface->abortRun(true);
    }
  EndOfRun r(aRun);
  m_endOfRunSignal(&r);
}

