#include "SimG4Core/Application/interface/RunAction.h"
#include "SimG4Core/Application/interface/RunManager.h"

#include "SimG4Core/Notification/interface/BeginOfRun.h"
#include "SimG4Core/Notification/interface/EndOfRun.h"
 
#include <iostream>
#include <fstream>
 
RunAction::RunAction(const edm::ParameterSet& p, RunManager* rm) 
   : m_runManager(rm), 
    m_stopFile(p.getParameter<std::string>("StopFile")) {}

void RunAction::BeginOfRunAction(const G4Run * aRun)
{
    if (std::ifstream(m_stopFile.c_str()))
    {
        std::cout << "BeginOfRunAction: termination signal received" << std::endl;
        //RunManager::instance()->abortRun(true);
	m_runManager->abortRun(true);
    }
    BeginOfRun r(aRun);
    m_beginOfRunSignal(&r);
}

void RunAction::EndOfRunAction(const G4Run * aRun)
{
    if (std::ifstream(m_stopFile.c_str()))
    {
        std::cout << "EndOfRunAction: termination signal received" << std::endl;
        //RunManager::instance()->abortRun(true);
	m_runManager->abortRun(true);
    }
    EndOfRun r(aRun);
    m_endOfRunSignal(&r);
}

