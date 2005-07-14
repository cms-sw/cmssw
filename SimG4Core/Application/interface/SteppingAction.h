#ifndef SimG4Core_SteppingAction_H
#define SimG4Core_SteppingAction_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "G4UserSteppingAction.hh"

class SteppingAction: public G4UserSteppingAction
{
public:
    //SteppingAction(const edm::ParameterSet & ps);
    SteppingAction();
    ~SteppingAction();
    void UserSteppingAction(const G4Step * aStep);
private:
    void catchLowEnergyInVacuumHere(const G4Step * aStep);
    void catchLowEnergyInVacuumNext(const G4Step * aStep);
private:
    bool   killBeamPipe;
    double theCriticalEnergyForVacuum;
    double theCriticalDensity;
    int    verbose;
};

#endif
