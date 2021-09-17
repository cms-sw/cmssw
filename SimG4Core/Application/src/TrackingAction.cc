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

//#define EDM_ML_DEBUG

TrackingAction::TrackingAction(EventAction* e, const edm::ParameterSet& p, CMSSteppingVerbose* sv)
    : eventAction_(e),
      currentTrack_(nullptr),
      steppingVerbose_(sv),
      g4Track_(nullptr),
      checkTrack_(p.getUntrackedParameter<bool>("CheckTrack", false)),
      doFineCalo_(p.getParameter<bool>("DoFineCalo")),
      saveCaloBoundaryInformation_(p.getParameter<bool>("SaveCaloBoundaryInformation")),
      eMinFine_(p.getParameter<double>("EminFineTrack") * CLHEP::MeV) {}

TrackingAction::~TrackingAction() {}

void TrackingAction::PreUserTrackingAction(const G4Track* aTrack) {
  g4Track_ = aTrack;
  currentTrack_ = new TrackWithHistory(aTrack);

  BeginOfTrack bt(aTrack);
  m_beginOfTrackSignal(&bt);

  TrackInformation* trkInfo = (TrackInformation*)aTrack->GetUserInformation();
  if (trkInfo && trkInfo->isPrimary()) {
    eventAction_->prepareForNewPrimary();
  }
  if (nullptr != steppingVerbose_) {
    steppingVerbose_->TrackStarted(aTrack, false);
  }

  if (doFineCalo_) {
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("DoFineCalo") << "PreUserTrackingAction: Start processing track " << aTrack->GetTrackID()
                                   << " pdgid=" << aTrack->GetDefinition()->GetPDGEncoding()
                                   << " ekin[GeV]=" << aTrack->GetKineticEnergy() / CLHEP::GeV << " vertex[cm]=("
                                   << aTrack->GetVertexPosition().x() / CLHEP::cm << ","
                                   << aTrack->GetVertexPosition().y() / CLHEP::cm << ","
                                   << aTrack->GetVertexPosition().z() / CLHEP::cm << ")"
                                   << " parentid=" << aTrack->GetParentID();
#endif
    // It is impossible to tell whether daughter tracks if this track may need to be saved at
    // this point; Therefore, *every* track is put in history, so that it can potentially be saved
    // later.
    trkInfo->putInHistory();
    // Always save primaries
    // Decays from primaries are marked as primaries (see NewTrackAction), but are not saved by
    // default. The primary is the earliest ancestor, and it must be saved.
    if (trkInfo->isPrimary())
      currentTrack_->save();
  }
}

void TrackingAction::PostUserTrackingAction(const G4Track* aTrack) {
  if (eventAction_->trackContainer() != nullptr) {
    uint32_t id = aTrack->GetTrackID();
    math::XYZVectorD pos(aTrack->GetStep()->GetPostStepPoint()->GetPosition().x(),
                         aTrack->GetStep()->GetPostStepPoint()->GetPosition().y(),
                         aTrack->GetStep()->GetPostStepPoint()->GetPosition().z());
    math::XYZTLorentzVectorD mom;
    std::pair<math::XYZVectorD, math::XYZTLorentzVectorD> p(pos, mom);

#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("DoFineCalo") << "PostUserTrackingAction:"
                                   << " aTrack->GetTrackID()=" << aTrack->GetTrackID()
                                   << " currentTrack_->saved()=" << currentTrack_->saved() << " PostStepPosition=("
                                   << pos.x() << "," << pos.y() << "," << pos.z() << ")";
#endif

    if (doFineCalo_) {
      TrackInformation* trkInfo = (TrackInformation*)aTrack->GetUserInformation();
      // Add the post-step position for _every_ track
      // in history to the TrackManager. Tracks in history _may_ be upgraded to stored
      // tracks, at which point the post-step position is needed again.
      eventAction_->addTkCaloStateInfo(id, p);
      if (trkInfo->crossedBoundary()) {
        currentTrack_->save();
        currentTrack_->setCrossedBoundaryPosMom(id, trkInfo->getPositionAtBoundary(), trkInfo->getMomentumAtBoundary());
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("DoFineCalo") << "PostUserTrackingAction:"
                                       << " Track " << aTrack->GetTrackID() << " crossed boundary; pos=("
                                       << trkInfo->getPositionAtBoundary().x() << ","
                                       << trkInfo->getPositionAtBoundary().y() << ","
                                       << trkInfo->getPositionAtBoundary().z() << ")"
                                       << " mom[GeV]=(" << trkInfo->getMomentumAtBoundary().x() << ","
                                       << trkInfo->getMomentumAtBoundary().y() << ","
                                       << trkInfo->getMomentumAtBoundary().z() << ","
                                       << trkInfo->getMomentumAtBoundary().e() << ")";
#endif
      }
    }

    TrackInformation* trkInfo = (TrackInformation*)aTrack->GetUserInformation();
    if (extractor_(aTrack).storeTrack() || currentTrack_->saved() ||
        (saveCaloBoundaryInformation_ && trkInfo->crossedBoundary())) {
      if (trkInfo->crossedBoundary()) {
        currentTrack_->setCrossedBoundaryPosMom(id, trkInfo->getPositionAtBoundary(), trkInfo->getMomentumAtBoundary());
      }
      currentTrack_->save();

      eventAction_->addTkCaloStateInfo(id, p);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("SimTrackManager")
          << "TrackingAction addTkCaloStateInfo " << id << " of momentum " << mom << " at " << pos;
#endif
    }

    bool withAncestor =
        ((extractor_(aTrack).getIDonCaloSurface() == aTrack->GetTrackID()) || (extractor_(aTrack).isAncestor()));

    if (extractor_(aTrack).isInHistory()) {
      // check with end-of-track information
      if (checkTrack_) {
        currentTrack_->checkAtEnd(aTrack);
      }

      eventAction_->addTrack(currentTrack_, true, withAncestor);

#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("SimTrackManager")
          << "TrackingAction addTrack " << currentTrack_->trackID() << " E(GeV)= " << aTrack->GetKineticEnergy() << "  "
          << aTrack->GetDefinition()->GetParticleName() << " added= " << withAncestor << " at "
          << aTrack->GetPosition();
      edm::LogVerbatim("SimTrackManager") << "TrackingAction addTrack " << currentTrack_->trackID() << " added with "
                                          << true << " and " << withAncestor << " at " << pos;
#endif

    } else {
      eventAction_->addTrack(currentTrack_, false, false);

#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("SimTrackManager")
          << "TrackingAction addTrack " << currentTrack_->trackID() << " added with " << false << " and " << false;
#endif

      delete currentTrack_;
    }
  }
  if (nullptr != steppingVerbose_) {
    steppingVerbose_->TrackEnded(aTrack);
  }

  EndOfTrack et(aTrack);
  m_endOfTrackSignal(&et);

  currentTrack_ = nullptr;  // reset for next track
}
