#ifndef SimG4Core_KillSecondariesRunAction_H
#define SimG4Core_KillSecondariesRunAction_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimG4Core/Notification/interface/Observer.h"
#include "SimG4Core/Watcher/interface/SimWatcher.h"

class BeginOfRun;

class KillSecondariesRunAction : public SimWatcher, public Observer<const BeginOfRun *> {
public:
  KillSecondariesRunAction(edm::ParameterSet const &p);
  ~KillSecondariesRunAction() override;
  void update(const BeginOfRun *run) override;
};

#endif
