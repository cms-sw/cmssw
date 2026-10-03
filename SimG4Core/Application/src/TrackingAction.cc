#include "SimG4Core/Application/interface/TrackingAction.h"
#include "SimG4Core/Physics/interface/CMSG4TrackInterface.h"

#include "SimG4Core/Notification/interface/BeginOfTrack.h"
#include "SimG4Core/Notification/interface/EndOfTrack.h"
#include "SimG4Core/Notification/interface/TrackInformation.h"
#include "SimG4Core/Notification/interface/TrackWithHistory.h"
#include "SimG4Core/Notification/interface/SimTrackManager.h"
#include "SimG4Core/Notification/interface/CMSSteppingVerbose.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4UImanager.hh"
#include "G4TrackingManager.hh"
#include <CLHEP/Units/SystemOfUnits.h>
#include <algorithm>

//#define EDM_ML_DEBUG

TrackingAction::TrackingAction(SimTrackManager* stm, CMSSteppingVerbose* sv, const edm::ParameterSet& p)
    : trackManager_(stm),
      steppingVerbose_(sv),
      endPrintTrackID_(p.getParameter<int>("EndPrintTrackID")),
      checkTrack_(p.getUntrackedParameter<bool>("CheckTrack", false)),
      doFineCalo_(p.getParameter<bool>("DoFineCalo")),
      saveCaloBoundaryInformation_(p.getParameter<bool>("SaveCaloBoundaryInformation")),
      ekinMin_(p.getParameter<double>("PersistencyEmin") * CLHEP::GeV),
      ekinMinRegion_(p.getParameter<std::vector<double>>("RegionEmin")) {
  interface_ = CMSG4TrackInterface::instance();
  // Keep the SimTrack/SimVertex history connected to the generator at a high
  // PersistencyEmin by reattaching dropped-parent vertices to the nearest stored
  // ancestor (no orphans), instead of lowering the threshold to keep everything.
  // Baseline TrackingAction parameter (default False); enabled under enableTruth.
  trackManager_->setReconnectDroppedAncestors(p.getParameter<bool>("ReconnectDroppedAncestors"));
  double eth = p.getParameter<double>("EminFineTrack") * CLHEP::MeV;
  if (doFineCalo_ && eth < ekinMin_) {
    ekinMin_ = eth;
  }
  edm::LogVerbatim("SimG4CoreApplication") << "TrackingAction: boundary: " << saveCaloBoundaryInformation_
                                           << "; DoFineCalo: " << doFineCalo_ << "; ekinMin(MeV)=" << ekinMin_;
  if (!ekinMinRegion_.empty()) {
    ptrRegion_.resize(ekinMinRegion_.size(), nullptr);
  }
}

void TrackingAction::PreUserTrackingAction(const G4Track* aTrack) {
  // a resumed track continues with its history, no new BeginOfTrack
  if (aTrack->GetCurrentStepNumber() > 0 && resumeSuspended(aTrack)) {
    return;
  }
  resumed_ = false;
  g4Track_ = aTrack;
  currentHistory_ = new TrackWithHistory(aTrack, aTrack->GetParentID());
  interface_->setCurrentTrack(aTrack);

  BeginOfTrack bt(aTrack);
  m_beginOfTrackSignal(&bt);

  trkInfo_ = dynamic_cast<TrackInformation*>(aTrack->GetUserInformation());

  // Always save primaries
  if (nullptr != trkInfo_ && trkInfo_->isPrimary()) {
    trackManager_->cleanTracksWithHistory();
    currentHistory_->setToBeSaved();
  }

  if (nullptr != steppingVerbose_) {
    steppingVerbose_->trackStarted(aTrack, false);
    if (aTrack->GetTrackID() == endPrintTrackID_) {
      steppingVerbose_->stopEventPrint();
    }
  }
  double ekin = aTrack->GetKineticEnergy();

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("DoFineCalo") << "PreUserTrackingAction: Start processing track " << aTrack->GetTrackID()
                                 << " pdgid=" << aTrack->GetDefinition()->GetPDGEncoding()
                                 << " ekin[GeV]=" << ekin / CLHEP::GeV << " vertex[cm]=("
                                 << aTrack->GetVertexPosition().x() / CLHEP::cm << ","
                                 << aTrack->GetVertexPosition().y() / CLHEP::cm << ","
                                 << aTrack->GetVertexPosition().z() / CLHEP::cm << ")"
                                 << " parentid=" << aTrack->GetParentID();
#endif
  if (nullptr != trkInfo_ && ekin > ekinMin_) {
    // Each track with energy above the threshold should be saved
    trkInfo_->putInHistory();
  }
}

void TrackingAction::PostUserTrackingAction(const G4Track* aTrack) {
  if (resumed_) {
    endResumed(aTrack);
    return;
  }
  // Tracks in history may be upgraded to stored secondary tracks,
  // which cross the boundary between Tracker and Calo
  int id = aTrack->GetTrackID();
  bool ok = currentHistory_->saved();
  if (nullptr != trkInfo_) {
    ok = (ok || trkInfo_->storeTrack());
    if (trkInfo_->crossedBoundary()) {
      currentHistory_->setCrossedBoundaryPosMom(
          id, trkInfo_->getPositionAtBoundary(), trkInfo_->getMomentumAtBoundary());
      ok = (ok || saveCaloBoundaryInformation_ || doFineCalo_);
    }
  }
  if (ok) {
    currentHistory_->setToBeSaved();
  }

  bool withAncestor = false;
  bool isInHistory = false;
  if (nullptr != trkInfo_) {
    withAncestor = (trkInfo_->getIDonCaloSurface() == id || trkInfo_->isAncestor());
    isInHistory = trkInfo_->isInHistory();
  }

  trackManager_->addTrack(currentHistory_, aTrack, isInHistory, withAncestor);

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("TrackingAction") << "TrackingAction end track=" << id << "  "
                                     << aTrack->GetDefinition()->GetParticleName() << " proposed to be saved= " << ok
                                     << " end point " << aTrack->GetPosition();
#endif

  // suspended: keep the history, the track ends later
  if (aTrack->GetTrackStatus() == fSuspend) {
    suspended_.push_back({id, currentHistory_, !isInHistory});
    return;
  }

  if (!isInHistory) {
    delete currentHistory_;
  }

  EndOfTrack et(aTrack);
  m_endOfTrackSignal(&et);
}

bool TrackingAction::resumeSuspended(const G4Track* aTrack) {
  const int id = aTrack->GetTrackID();
  auto itr = std::find_if(suspended_.begin(), suspended_.end(), [id](const SuspendedTrack& s) { return s.id == id; });
  if (itr == suspended_.end()) {
    return false;
  }
  g4Track_ = aTrack;
  currentHistory_ = itr->history;
  owned_ = itr->owned;
  suspended_.erase(itr);
  resumed_ = true;
  interface_->setCurrentTrack(aTrack);
  trkInfo_ = dynamic_cast<TrackInformation*>(aTrack->GetUserInformation());
  return true;
}

void TrackingAction::endResumed(const G4Track* aTrack) {
  // history already given to the track manager at the first suspension
  const int id = aTrack->GetTrackID();
  if (nullptr != trkInfo_) {
    if (trkInfo_->storeTrack()) {
      currentHistory_->setToBeSaved();
    }
    if (trkInfo_->crossedBoundary()) {
      currentHistory_->setCrossedBoundaryPosMom(
          id, trkInfo_->getPositionAtBoundary(), trkInfo_->getMomentumAtBoundary());
      if (saveCaloBoundaryInformation_ || doFineCalo_) {
        currentHistory_->setToBeSaved();
      }
    }
  }
  if (aTrack->GetTrackStatus() == fSuspend) {
    suspended_.push_back({id, currentHistory_, owned_});
    return;
  }
  resumed_ = false;
  if (owned_) {
    delete currentHistory_;
  }
  EndOfTrack et(aTrack);
  m_endOfTrackSignal(&et);
}
