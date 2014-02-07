#include "SimG4Core/Application/interface/RunAction.h"
#include "SimG4Core/Application/interface/SimRunInterface.h"

#include "SimG4Core/Notification/interface/BeginOfRun.h"
#include "SimG4Core/Notification/interface/EndOfRun.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
 
#include <iostream>
#include <fstream>
 
RunAction::RunAction(const edm::ParameterSet& p, SimRunInterface* rm) 
   : m_runInterface(rm), 
     m_stopFile(p.getParameter<std::string>("StopFile")) {}

void RunAction::BeginOfRunAction(const G4Run * aRun)
{
  if (std::ifstream(m_stopFile.c_str()))
    {
      edm::LogWarning("SimG4CoreApplication")
        << "BeginOfRunAction: termination signal received";
      //std::cout << "BeginOfRunAction: termination signal received" << std::endl;
      m_runInterface->abortRun(true);
    }
    BeginOfRun r(aRun);
    m_beginOfRunSignal(&r);
}

void RunAction::EndOfRunAction(const G4Run * aRun)
{
  if (std::ifstream(m_stopFile.c_str()))
    {
      edm::LogWarning("SimG4CoreApplication")
        << "EndOfRunAction: termination signal received";
      //std::cout << "EndOfRunAction: termination signal received" << std::endl;
      m_runInterface->abortRun(true);
    }
  EndOfRun r(aRun);
  m_endOfRunSignal(&r);
}

