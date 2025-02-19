#ifndef SimG4Core_KillSecondariesTrackAction_H
#define SimG4Core_KillSecondariesTrackAction_H

#include "SimG4Core/Watcher/interface/SimWatcher.h"
#include "SimG4Core/Notification/interface/Observer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
   
class BeginOfTrack;

class KillSecondariesTrackAction : public SimWatcher,
  public Observer<const BeginOfTrack *> {

public:
  KillSecondariesTrackAction(edm::ParameterSet const & p);
  ~KillSecondariesTrackAction();
  void update(const BeginOfTrack * trk);

private:
  bool   killHeavy;
  double kmaxIon, kmaxNeutron, kmaxProton;
};

#endif


