#ifndef SimG4Core_RunAction_H
#define SimG4Core_RunAction_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "G4UserRunAction.hh"

#include <string>
#include "boost/signal.hpp"

class RunManager;
class BeginOfRun;
class EndOfRun;

class RunAction: public G4UserRunAction
{
public:
    RunAction(const edm::ParameterSet & ps);
    void BeginOfRunAction(const G4Run * aRun);
    void EndOfRunAction(const G4Run * aRun);
    
    boost::signal< void(const BeginOfRun*)> m_beginOfRunSignal;
    boost::signal< void(const EndOfRun*)> m_endOfRunSignal;
private:
    std::string m_stopFile;
};

#endif
