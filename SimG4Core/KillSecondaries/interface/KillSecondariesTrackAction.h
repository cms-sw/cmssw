#ifndef SimG4Core_KillSecondariesTrackAction_H
#define SimG4Core_KillSecondariesTrackAction_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimG4Core/Notification/interface/Observer.h"
#include "SimG4Core/Watcher/interface/SimWatcher.h"

class BeginOfTrack;

class KillSecondariesTrackAction : public SimWatcher, public Observer<const BeginOfTrack *> {
public:
  KillSecondariesTrackAction(edm::ParameterSet const &p);
  ~KillSecondariesTrackAction() override;
  void update(const BeginOfTrack *trk) override;

private:
  bool killHeavy;
  double kmaxIon, kmaxNeutron, kmaxProton;
};

#endif
