#ifndef SimG4Core_TrackingAction_H
#define SimG4Core_TrackingAction_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimG4Core/Notification/interface/SimActivityRegistry.h"

#include "G4UserTrackingAction.hh"
#include "G4VSolid.hh"

class EventAction;
class TrackWithHistory; 
class BeginOfTrack;
class EndOfTrack;

class TrackingAction : public G4UserTrackingAction
{
public:
    TrackingAction(EventAction * ea, const edm::ParameterSet & ps);
    virtual ~TrackingAction();
    virtual void PreUserTrackingAction(const G4Track * aTrack);
    virtual void PostUserTrackingAction(const G4Track * aTrack);
    TrackWithHistory * currentTrackWithHistory() { return currentTrack_; }
    G4TrackingManager * getTrackManager();

    SimActivityRegistry::BeginOfTrackSignal m_beginOfTrackSignal;
    SimActivityRegistry::EndOfTrackSignal m_endOfTrackSignal;
private:
    bool isNewPrimary(const G4Track * aTrack);
private:
    EventAction * eventAction_;
    TrackWithHistory * currentTrack_;
    G4VSolid * worldSolid;
    bool detailedTiming;
    int  trackMgrVerbose;
};

#endif
