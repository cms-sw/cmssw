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
      steppingVerbose_(sv),
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
    delete currentTrack_;
  }
  if (nullptr != steppingVerbose_) {
    steppingVerbose_->TrackEnded(aTrack);
  }

  EndOfTrack et(aTrack);
  m_endOfTrackSignal(&et);
}
