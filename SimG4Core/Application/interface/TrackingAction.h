#ifndef SimG4Core_TrackingAction_H
#define SimG4Core_TrackingAction_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "G4UserTrackingAction.hh"

class EventAction;
class TrackWithHistory; 

class TrackingAction : public G4UserTrackingAction
{
public:
    explicit TrackingAction(EventAction * ea, const edm::ParameterSet & ps);
    virtual ~TrackingAction();
    virtual void PreUserTrackingAction(const G4Track * aTrack);
    virtual void PostUserTrackingAction(const G4Track * aTrack);
    TrackWithHistory * currentTrackWithHistory() { return currentTrack_; }
private:
    EventAction * eventAction_;
    TrackWithHistory * currentTrack_;
    bool detailedTiming;
};

#endif
