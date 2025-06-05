#ifndef SimG4Core_Phase2TrackingAction_H
#define SimG4Core_Phase2TrackingAction_H
//
// Package:     Application
// Class  :     Phase2TrackingAction
//
// Description: Manage tracks
// Created:     08.04.2025 V.Ivantchenko
//

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimG4Core/Notification/interface/SimActivityRegistry.h"

#include "G4UserTrackingAction.hh"
#include "G4Region.hh"

#include <vector>

class SimTrackManager;
class CMSG4TrackInterface;
class TrackWithHistory;
class BeginOfTrack;
class EndOfTrack;
class CMSSteppingVerbose;
class TrackInformation;

class Phase2TrackingAction : public G4UserTrackingAction {
public:
  explicit Phase2TrackingAction(SimTrackManager*, CMSSteppingVerbose*, const edm::ParameterSet& ps);
  ~Phase2TrackingAction() override = default;

  void PreUserTrackingAction(const G4Track* aTrack) override;
  void PostUserTrackingAction(const G4Track* aTrack) override;

  inline TrackWithHistory* currentTrackWithHistory() { return currentTrack_; }
  inline const G4Track* geant4Track() const { return g4Track_; }
  inline G4TrackingManager* getTrackManager() { return fpTrackingManager; }

  SimActivityRegistry::BeginOfTrackSignal m_beginOfTrackSignal;
  SimActivityRegistry::EndOfTrackSignal m_endOfTrackSignal;

  Phase2TrackingAction(Phase2TrackingAction&) = delete;
  Phase2TrackingAction& operator=(const Phase2TrackingAction& right) = delete;

private:
  SimTrackManager* trackManager_;
  CMSG4TrackInterface* trackInterface_;
  CMSSteppingVerbose* steppingVerbose_;
  const G4Track* g4Track_ = nullptr;
  TrackInformation* trkInfo_ = nullptr;
  TrackWithHistory* currentTrack_ = nullptr;
  int endPrintTrackID_;
  bool checkTrack_;
  bool doFineCalo_;
  bool saveCaloBoundaryInformation_;
  double ekinMin_;
  std::vector<double> ekinMinRegion_;
  std::vector<G4Region*> ptrRegion_;
};

#endif
