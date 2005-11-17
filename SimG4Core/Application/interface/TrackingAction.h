#ifndef SimG4Core_TrackingAction_H
#define SimG4Core_TrackingAction_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "G4UserTrackingAction.hh"
#include "boost/signal.hpp"

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

    boost::signal< void(const BeginOfTrack*)> m_beginOfTrackSignal;
    boost::signal< void(const EndOfTrack*)> m_endOfTrackSignal;

private:
    EventAction * eventAction_;
    TrackWithHistory * currentTrack_;
    bool detailedTiming;
};

#endif
