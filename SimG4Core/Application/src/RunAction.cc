#include "SimG4Core/Application/interface/RunAction.h"
#include "SimG4Core/Application/interface/SimRunInterface.h"

#include "SimG4Core/Notification/interface/BeginOfRun.h"
#include "SimG4Core/Notification/interface/EndOfRun.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>
#include <fstream>

RunAction::RunAction(const edm::ParameterSet& p, SimRunInterface* rm, bool) 
  : m_runInterface(rm), m_stopFile(p.getParameter<std::string>("StopFile"))
{}

RunAction::~RunAction()
{}

void RunAction::BeginOfRunAction(const G4Run * aRun)
{
  if (!m_stopFile.empty() && std::ifstream(m_stopFile.c_str()))
    {
      edm::LogWarning("SimG4CoreApplication")
        << "RunAction::BeginOfRunAction: termination signal received";
      m_runInterface->abortRun(true);
    }
  BeginOfRun r(aRun);
  m_beginOfRunSignal(&r);
}

void RunAction::EndOfRunAction(const G4Run * aRun)
{
  EndOfRun r(aRun);
  m_endOfRunSignal(&r);
}

