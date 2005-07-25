#ifndef SimG4CMS_TkSimTrackSelection_H
#define SimG4CMS_TkSimTrackSelection_H

#include "SimG4Core/Notification/interface/BeginOfTrack.h"
#include "SimG4Core/Notification/interface/EndOfTrack.h"

class TrackInformation;
/**
 * Selects the G4Tracks which should be made persistent
 */

class TkSimTrackSelection
{
public:
    TkSimTrackSelection();
    void upDate(const BeginOfTrack *);
private:
    TrackInformation * getOrCreateTrackInformation(const G4Track *);
    float energyCut;
    float energyHistoryCut;
    // definition of Tracker volume
    float rTracker;
    float zTracker;
};

#endif
