#ifndef SimG4Core_TrackingAction_H
#define SimG4Core_TrackingAction_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimG4Core/Notification/interface/SimActivityRegistry.h"

#include "G4UserTrackingAction.hh"

class SimTrackManager;
class TrackWithHistory;
class BeginOfTrack;
class EndOfTrack;
class CMSSteppingVerbose;
class TrackInformation;

class TrackingAction : public G4UserTrackingAction {
public:
  explicit TrackingAction(SimTrackManager*, CMSSteppingVerbose*, const edm::ParameterSet& ps);
  ~TrackingAction() override = default;

  void PreUserTrackingAction(const G4Track* aTrack) override;
  void PostUserTrackingAction(const G4Track* aTrack) override;

  inline TrackWithHistory* currentTrackWithHistory() { return currentTrack_; }
  inline const G4Track* geant4Track() const { return g4Track_; }
  inline G4TrackingManager* getTrackManager() { return fpTrackingManager; }

  SimActivityRegistry::BeginOfTrackSignal m_beginOfTrackSignal;
  SimActivityRegistry::EndOfTrackSignal m_endOfTrackSignal;

private:
  SimTrackManager* trackManager_;
  CMSSteppingVerbose* steppingVerbose_;
  const G4Track* g4Track_ = nullptr;
  TrackInformation* trkInfo_ = nullptr;
  TrackWithHistory* currentTrack_ = nullptr;
  int endPrintTrackID_;
  bool checkTrack_;
  bool doFineCalo_;
  bool saveCaloBoundaryInformation_;
  double eMinFine_;
};

#endif
