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

class TrackingAction : public G4UserTrackingAction {
public:
  explicit TrackingAction(EventAction* ea, const edm::ParameterSet& ps, CMSSteppingVerbose*);
  ~TrackingAction() override;

  void PreUserTrackingAction(const G4Track* aTrack) override;
  void PostUserTrackingAction(const G4Track* aTrack) override;

  inline TrackWithHistory* currentTrackWithHistory() { return currentTrack_; }
  inline const G4Track* geant4Track() const { return g4Track_; }
  inline G4TrackingManager* getTrackManager() { return fpTrackingManager; }

  SimActivityRegistry::BeginOfTrackSignal m_beginOfTrackSignal;
  SimActivityRegistry::EndOfTrackSignal m_endOfTrackSignal;

private:
  TrackInformationExtractor extractor_;
  EventAction* eventAction_;
  TrackWithHistory* currentTrack_;
  CMSSteppingVerbose* steppingVerbose_;
  const G4Track* g4Track_;

  // suspended tracks (fSuspend) keep their first history until they end
  struct SuspendedTrack {
    int id;
    TrackWithHistory* history;
    bool owned;
  };
  bool resumeSuspended(const G4Track* aTrack);
  void endResumed(const G4Track* aTrack);
  std::vector<SuspendedTrack> suspended_;
  bool quirkTracking_;
  bool resumed_ = false;
  bool owned_ = false;
  bool checkTrack_;
  bool doFineCalo_;
  bool saveCaloBoundaryInformation_;
  double eMinFine_;
};

#endif
