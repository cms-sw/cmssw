#ifndef SimG4Core_SteppingAction_H
#define SimG4Core_SteppingAction_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimG4Core/Notification/interface/SimActivityRegistry.h"

#include "G4UserSteppingAction.hh"

class EventAction;

class SteppingAction: public G4UserSteppingAction
{
public:
    SteppingAction(EventAction * ea,const edm::ParameterSet & ps);
    ~SteppingAction();
    void UserSteppingAction(const G4Step * aStep);

    SimActivityRegistry::G4StepSignal m_g4StepSignal;
private:
    void catchLowEnergyInVacuumHere(const G4Step * aStep);
    void catchLowEnergyInVacuumNext(const G4Step * aStep);
    void storeTkCaloStateInfo(const G4Step * aStep);
private:
    EventAction * eventAction_;
    bool   killBeamPipe;
    double theCriticalEnergyForVacuum;
    double theCriticalDensity;
    int    verbose;
};

#endif
