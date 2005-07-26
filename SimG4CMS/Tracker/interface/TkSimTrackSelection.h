#ifndef TrackerSim_TkSimTrackSelection_H
#define TrackerSim_TkSimTrackSelection_H

#include "Utilities/Notification/interface/Observer.h"
#include "SimG4Core/Notification/interface/BeginOfTrack.h"
#include "SimG4Core/Notification/interface/EndOfTrack.h"

class TrackInformation;
/**
 * Selects the G4Tracks which should be made persistent
 */

class TkSimTrackSelection :  private Observer<const BeginOfTrack *> {
  public:
  TkSimTrackSelection();
  void upDate(const BeginOfTrack *);
 private:
  TrackInformation* getOrCreateTrackInformation(const G4Track *);
  float energyCut;
  float energyHistoryCut;
  //
  // definition of Tracker volume
  //
  float rTracker;
  float zTracker;
};


#endif
