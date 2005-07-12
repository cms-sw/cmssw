#ifndef SimG4Core_RunAction_H
#define SimG4Core_RunAction_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "G4UserRunAction.hh"

#include <string>

class RunManager;

class RunAction: public G4UserRunAction
{
public:
    RunAction(const edm::ParameterSet & ps);
    void BeginOfRunAction(const G4Run * aRun);
    void EndOfRunAction(const G4Run * aRun);
private:
    std::string m_stopFile;
};

#endif
