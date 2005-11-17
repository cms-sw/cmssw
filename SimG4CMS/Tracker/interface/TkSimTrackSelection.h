#ifndef SimG4CMS_TkSimTrackSelection_H
#define SimG4CMS_TkSimTrackSelection_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimG4Core/Notification/interface/Observer.h"
#include "SimG4Core/Notification/interface/BeginOfTrack.h"
#include "SimG4Core/Notification/interface/EndOfTrack.h"

class TrackInformation;
/**
 * Selects the G4Tracks which should be made persistent
 */


class TkSimTrackSelection : public Observer<BeginOfTrack>{
  //public Observer<const BeginOfTrack*>{
  public:
  TkSimTrackSelection( edm::ParameterSet const & p);
  //  void update(const BeginOfTrack *) const;
  void update(const BeginOfTrack *);
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
