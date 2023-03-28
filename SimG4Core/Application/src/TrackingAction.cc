#include "SimG4Core/Application/interface/TrackingAction.h"

#include "SimG4Core/Notification/interface/NewTrackAction.h"
#include "SimG4Core/Notification/interface/CurrentG4Track.h"
#include "SimG4Core/Notification/interface/BeginOfTrack.h"
#include "SimG4Core/Notification/interface/EndOfTrack.h"
#include "SimG4Core/Notification/interface/TrackInformation.h"
#include "SimG4Core/Notification/interface/TrackWithHistory.h"
#include "SimG4Core/Notification/interface/SimTrackManager.h"
#include "SimG4Core/Notification/interface/CMSSteppingVerbose.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4UImanager.hh"
#include "G4TrackingManager.hh"
#include "G4SystemOfUnits.hh"

//#define EDM_ML_DEBUG

TrackingAction::TrackingAction(SimTrackManager* stm, CMSSteppingVerbose* sv, const edm::ParameterSet& p)
    : trackManager_(stm),
      steppingVerbose_(sv),
      endPrintTrackID_(p.getParameter<int>("EndPrintTrackID")),
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
    trackManager_->cleanTracksWithHistory();
    currentTrack_->setToBeSaved();
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
  if (ekin > eMinFine_) {
    // It is impossible to tell whether daughter tracks if this track may need to be saved at
    // this point; Therefore, every track above the threshold is put in history,
    // so that it can potentially be saved later.
    trkInfo_->putInHistory();
  }
}

void TrackingAction::PostUserTrackingAction(const G4Track* aTrack) {
  // Add the post-step position for every track in history to the TrackManager.
  // Tracks in history may be upgraded to stored tracks, at which point
  // the post-step position is needed again.
  int id = aTrack->GetTrackID();
  bool ok = (trkInfo_->storeTrack() || currentTrack_->saved());
  if (trkInfo_->crossedBoundary()) {
    currentTrack_->setCrossedBoundaryPosMom(id, trkInfo_->getPositionAtBoundary(), trkInfo_->getMomentumAtBoundary());
    ok = (ok || saveCaloBoundaryInformation_ || doFineCalo_);
  }
  if (ok) {
    currentTrack_->setToBeSaved();
  }

  bool withAncestor = (trkInfo_->getIDonCaloSurface() == id || trkInfo_->isAncestor());
  bool isInHistory = trkInfo_->isInHistory();

  trackManager_->addTrack(currentTrack_, aTrack, isInHistory, withAncestor);

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("TrackingAction") << "TrackingAction end track=" << id << "  "
                                     << aTrack->GetDefinition()->GetParticleName() << " ansestor= " << withAncestor
                                     << " saved= " << currentTrack_->saved() << " end point " << aTrack->GetPosition();
#endif

  EndOfTrack et(aTrack);
  m_endOfTrackSignal(&et);
}
