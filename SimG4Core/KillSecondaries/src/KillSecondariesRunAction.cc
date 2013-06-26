#include "SimG4Core/KillSecondaries/interface/KillSecondariesRunAction.h"
#include "SimG4Core/KillSecondaries/interface/KillSecondariesStackingAction.h"

#include "SimG4Core/Notification/interface/BeginOfRun.h"

#include "G4Run.hh"
#include "G4RunManagerKernel.hh"

KillSecondariesRunAction::KillSecondariesRunAction(edm::ParameterSet const & p)
    {}

KillSecondariesRunAction::~KillSecondariesRunAction() {}
 
void KillSecondariesRunAction::update(const BeginOfRun * r)
{ 
    std::cout << " Using KillSecondariesStackingAction!!! " << std::endl;
    G4RunManagerKernel::GetRunManagerKernel()->GetEventManager()->
	SetUserAction(new KillSecondariesStackingAction);
}

