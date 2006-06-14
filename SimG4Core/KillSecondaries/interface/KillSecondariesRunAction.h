#ifndef SimG4Core_KillSecondariesRunAction_H
#define SimG4Core_KillSecondariesRunAction_H

#include "SimG4Core/UtilityAction/interface/UtilityAction.h"
#include "SimG4Core/Notification/interface/Observer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
   
class BeginOfRun;

class KillSecondariesRunAction : public UtilityAction,
				 public Observer<const BeginOfRun *> 
{
public:
    KillSecondariesRunAction(edm::ParameterSet const & p);
    ~KillSecondariesRunAction();
    void update(const BeginOfRun * run);
};

#endif


