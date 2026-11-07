#include "SimG4Core/Application/interface/TrackingAction.h"
#include "SimG4Core/Application/interface/EventAction.h"
#include "SimG4Core/Notification/interface/NewTrackAction.h"
#include "SimG4Core/Notification/interface/CurrentG4Track.h"
#include "SimG4Core/Notification/interface/BeginOfTrack.h"
#include "SimG4Core/Notification/interface/EndOfTrack.h"
#include "SimG4Core/Notification/interface/TrackInformation.h"
#include "SimG4Core/Notification/interface/TrackWithHistory.h"
#include "SimG4Core/Notification/interface/CMSSteppingVerbose.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4UImanager.hh"
#include "G4TrackingManager.hh"
#include "G4SystemOfUnits.hh"
#include <algorithm>

//#define EDM_ML_DEBUG

TrackingAction::TrackingAction(EventAction* e, const edm::ParameterSet& p, CMSSteppingVerbose* sv)
    : eventAction_(e),
      steppingVerbose_(sv),
      quirkTracking_(p.getParameter<bool>("QuirkTracking")),
      checkTrack_(p.getUntrackedParameter<bool>("CheckTrack", false)),
      doFineCalo_(p.getParameter<bool>("DoFineCalo")),
      saveCaloBoundaryInformation_(p.getParameter<bool>("SaveCaloBoundaryInformation")),
      eMinFine_(p.getParameter<double>("EminFineTrack") * CLHEP::MeV) {
  if (!doFineCalo_) {
    eMinFine_ = DBL_MAX;
  }
  edm::LogVerbatim("SimG4CoreApplication") << "TrackingAction: boundary: " << saveCaloBoundaryInformation_
                                           << "; DoFineCalo: " << doFineCalo_ << "; EminFineTrack(MeV)=" << eMinFine_;
}

TrackingAction::~TrackingAction() {}

void TrackingAction::PreUserTrackingAction(const G4Track* aTrack) {
  // a resumed track continues with its history, no new BeginOfTrack
  if (quirkTracking_ && aTrack->GetCurrentStepNumber() > 0 && resumeSuspended(aTrack)) {
    return;
  }
  resumed_ = false;
  g4Track_ = aTrack;
  currentTrack_ = new TrackWithHistory(aTrack);

  BeginOfTrack bt(aTrack);
  m_beginOfTrackSignal(&bt);

  trkInfo_ = static_cast<TrackInformation*>(aTrack->GetUserInformation());

  // Always save primaries
  // Decays from primaries are marked as primaries (see NewTrackAction), but are not saved by
  // default. The primary is the earliest ancestor, and it must be saved.
  if (trkInfo_->isPrimary()) {
    eventAction_->prepareForNewPrimary();
    currentTrack_->save();
  }
  if (nullptr != steppingVerbose_) {
    steppingVerbose_->TrackStarted(aTrack, false);
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
  if (ekin > eMinFine_) {
    // It is impossible to tell whether daughter tracks if this track may need to be saved at
    // this point; Therefore, every track above the threshold is put in history,
    // so that it can potentially be saved later.
    trkInfo_->putInHistory();
  }
}

void TrackingAction::PostUserTrackingAction(const G4Track* aTrack) {
  if (resumed_) {
    endResumed(aTrack);
    return;
  }
  // suspended: keep the history, the track ends later
  const bool suspended = quirkTracking_ && aTrack->GetTrackStatus() == fSuspend;
  int id = aTrack->GetTrackID();
  const auto& ppos = aTrack->GetStep()->GetPostStepPoint()->GetPosition();
  math::XYZVectorD pos(ppos.x(), ppos.y(), ppos.z());
  math::XYZTLorentzVectorD mom;
  std::pair<math::XYZVectorD, math::XYZTLorentzVectorD> p(pos, mom);

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("DoFineCalo") << "PostUserTrackingAction:"
                                 << " aTrack->GetTrackID()=" << id
                                 << " currentTrack_->saved()=" << currentTrack_->saved();
#endif

  bool boundary = false;
  if (doFineCalo_) {
    // Add the post-step position for _every_ track
    // in history to the TrackManager. Tracks in history _may_ be upgraded to stored
    // tracks, at which point the post-step position is needed again.
    eventAction_->addTkCaloStateInfo(id, p);
    boundary = true;
  } else if (trkInfo_->storeTrack() || currentTrack_->saved() ||
             (saveCaloBoundaryInformation_ && trkInfo_->crossedBoundary())) {
    currentTrack_->save();
    eventAction_->addTkCaloStateInfo(id, p);
    boundary = true;
  }
  if (boundary && trkInfo_->crossedBoundary()) {
    currentTrack_->setCrossedBoundaryPosMom(id, trkInfo_->getPositionAtBoundary(), trkInfo_->getMomentumAtBoundary());
    currentTrack_->save();

#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("DoFineCalo") << "PostUserTrackingAction:"
                                   << " Track " << id << " crossed boundary; pos=("
                                   << trkInfo_->getPositionAtBoundary().x() << ","
                                   << trkInfo_->getPositionAtBoundary().y() << ","
                                   << trkInfo_->getPositionAtBoundary().z() << ")"
                                   << " mom[GeV]=(" << trkInfo_->getMomentumAtBoundary().x() << ","
                                   << trkInfo_->getMomentumAtBoundary().y() << ","
                                   << trkInfo_->getMomentumAtBoundary().z() << ","
                                   << trkInfo_->getMomentumAtBoundary().e() << ")";
#endif
  }

  bool withAncestor = (trkInfo_->getIDonCaloSurface() == id || trkInfo_->isAncestor());

  if (trkInfo_->isInHistory()) {
    // check with end-of-track information
    if (checkTrack_) {
      currentTrack_->checkAtEnd(aTrack);
    }

    eventAction_->addTrack(currentTrack_, true, withAncestor);

#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("SimTrackManager") << "TrackingAction addTrack " << id << "  "
                                        << aTrack->GetDefinition()->GetParticleName() << " added= " << withAncestor
                                        << " at " << aTrack->GetPosition();
#endif

  } else {
    eventAction_->addTrack(currentTrack_, false, false);

#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("SimTrackManager") << "TrackingAction addTrack " << id << " added with " << false << " and "
                                        << false;
#endif
    if (!suspended) {
      delete currentTrack_;
    }
  }
  if (suspended) {
    suspended_.push_back({id, currentTrack_, !trkInfo_->isInHistory()});
    return;
  }
  if (nullptr != steppingVerbose_) {
    steppingVerbose_->TrackEnded(aTrack);
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
  currentTrack_ = itr->history;
  owned_ = itr->owned;
  suspended_.erase(itr);
  resumed_ = true;
  trkInfo_ = dynamic_cast<TrackInformation*>(aTrack->GetUserInformation());
  return true;
}

void TrackingAction::endResumed(const G4Track* aTrack) {
  // history already given to the event action at the first suspension
  const int id = aTrack->GetTrackID();
  if (nullptr != trkInfo_) {
    if (trkInfo_->storeTrack()) {
      currentTrack_->save();
    }
    if (trkInfo_->crossedBoundary()) {
      currentTrack_->setCrossedBoundaryPosMom(id, trkInfo_->getPositionAtBoundary(), trkInfo_->getMomentumAtBoundary());
      if (saveCaloBoundaryInformation_ || doFineCalo_) {
        currentTrack_->save();
      }
    }
  }
  if (quirkTracking_ && aTrack->GetTrackStatus() == fSuspend) {
    suspended_.push_back({id, currentTrack_, owned_});
    return;
  }
  resumed_ = false;
  if (owned_) {
    delete currentTrack_;
  }
  if (nullptr != steppingVerbose_) {
    steppingVerbose_->TrackEnded(aTrack);
  }
  EndOfTrack et(aTrack);
  m_endOfTrackSignal(&et);
}
