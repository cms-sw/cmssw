#ifndef SimG4Core_TrackingAction_H
#define SimG4Core_TrackingAction_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimG4Core/Notification/interface/SimActivityRegistry.h"
#include "SimG4Core/Notification/interface/TrackInformationExtractor.h"

#include "G4UserTrackingAction.hh"

class EventAction;
class TrackWithHistory; 
class BeginOfTrack;
class EndOfTrack;
class CMSSteppingVerbose;

class TrackingAction : public G4UserTrackingAction
{
public:
    explicit TrackingAction(EventAction * ea, const edm::ParameterSet & ps, 
                            CMSSteppingVerbose*);
    virtual ~TrackingAction();

    virtual void PreUserTrackingAction(const G4Track * aTrack);
    virtual void PostUserTrackingAction(const G4Track * aTrack);

    inline TrackWithHistory* currentTrackWithHistory() { return currentTrack_; }
    inline const G4Track* geant4Track() const { return g4Track_; }
    inline G4TrackingManager * getTrackManager() { return fpTrackingManager; }

    SimActivityRegistry::BeginOfTrackSignal m_beginOfTrackSignal;
    SimActivityRegistry::EndOfTrackSignal m_endOfTrackSignal;

private:

    TrackInformationExtractor extractor_;
    EventAction * eventAction_;
    TrackWithHistory * currentTrack_;
    CMSSteppingVerbose* steppingVerbose_;
    const G4Track * g4Track_;
    bool checkTrack_;
};

#endif
