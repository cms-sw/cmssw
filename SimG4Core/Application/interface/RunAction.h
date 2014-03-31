#ifndef SimG4Core_RunAction_H
#define SimG4Core_RunAction_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimG4Core/Notification/interface/SimActivityRegistry.h"

#include "G4UserRunAction.hh"

#include <string>

class SimRunInterface;
class BeginOfRun;
class EndOfRun;

class RunAction: public G4UserRunAction
{
public:
    RunAction(const edm::ParameterSet & ps, SimRunInterface*);
    void BeginOfRunAction(const G4Run * aRun);
    void EndOfRunAction(const G4Run * aRun);
    
    SimActivityRegistry::BeginOfRunSignal m_beginOfRunSignal;
    SimActivityRegistry::EndOfRunSignal m_endOfRunSignal; 
private:
    SimRunInterface* m_runInterface;
    std::string m_stopFile;
};

#endif
