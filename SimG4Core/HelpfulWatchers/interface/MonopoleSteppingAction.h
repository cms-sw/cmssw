#ifndef SimG4Core_HelpfulWatchers_MonopoleSteppingAction_H
#define SimG4Core_HelpfulWatchers_MonopoleSteppingAction_H

#include "SimG4Core/Watcher/interface/SimWatcher.h"
#include "SimG4Core/Notification/interface/Observer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "G4Step.hh"
#include <vector>

class BeginOfJob;
class BeginOfRun;
class BeginOfTrack;

class MonopoleSteppingAction : public SimWatcher,
                               public Observer<const BeginOfJob *>,
                               public Observer<const BeginOfRun *>,
                               public Observer<const BeginOfTrack *>,
                               public Observer<const G4Step *> {
public:
  MonopoleSteppingAction(edm::ParameterSet const & p);
  ~MonopoleSteppingAction();
  void update(const BeginOfJob*);
  void update(const BeginOfRun*);
  void update(const BeginOfTrack*);
  void update(const G4Step*);

private:
  bool             mode, actOnTrack;
  std::vector<int> pdgCode;
  double           eStart, pxStart, pyStart, pzStart;
  double           dirxStart, diryStart, dirzStart;
  double           cMevToJ, cMeVToKgMByS, cInMByS, magCharge, bZ;
};

#endif
