#include "SimG4Core/Application/interface/Phase2TrackingAction.h"
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

//#define EDM_ML_DEBUG

Phase2TrackingAction::Phase2TrackingAction(SimTrackManager* stm,
					   CMSSteppingVerbose* sv, const edm::ParameterSet& p)
    : trackManager_(stm),
      steppingVerbose_(sv),
      endPrintTrackID_(p.getParameter<int>("EndPrintTrackID")),
      checkTrack_(p.getUntrackedParameter<bool>("CheckTrack", false)),
      doFineCalo_(p.getParameter<bool>("DoFineCalo")),
      saveCaloBoundaryInformation_(p.getParameter<bool>("SaveCaloBoundaryInformation")),
      ekinMin_(p.getParameter<double>("PersistencyEmin") * CLHEP::GeV),
      ekinMinRegion_(p.getParameter<std::vector<double>>("RegionEmin")) {
  trackInterface_ = CMSG4TrackInterface::instance();
  double eth = p.getParameter<double>("EminFineTrack") * CLHEP::MeV;
  if (doFineCalo_ && eth < ekinMin_) {
    ekinMin_ = eth;
  }
  edm::LogVerbatim("SimG4CoreApplication")
      << "Phase2TrackingAction: boundary: " << saveCaloBoundaryInformation_
      << "; DoFineCalo: " << doFineCalo_ << "; ekinMin(MeV)=" << ekinMin_;
  if (!ekinMinRegion_.empty()) {
    ptrRegion_.resize(ekinMinRegion_.size(), nullptr);
  }
}

void Phase2TrackingAction::PreUserTrackingAction(const G4Track* aTrack) {
  g4Track_ = aTrack;
  currentTrack_ = new TrackWithHistory(aTrack, aTrack->GetParentID());
  trackInterface_->setCurrentTrack(aTrack);

  BeginOfTrack bt(aTrack);
  m_beginOfTrackSignal(&bt);

  trkInfo_ = static_cast<TrackInformation*>(aTrack->GetUserInformation());

  // Always save primaries
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
  edm::LogVerbatim("DoFineCalo") << "PreUserPhase2TrackingAction: Start processing track " << aTrack->GetTrackID()
                                 << " pdgid=" << aTrack->GetDefinition()->GetPDGEncoding()
                                 << " ekin[GeV]=" << ekin / CLHEP::GeV << " vertex[cm]=("
                                 << aTrack->GetVertexPosition().x() / CLHEP::cm << ","
                                 << aTrack->GetVertexPosition().y() / CLHEP::cm << ","
                                 << aTrack->GetVertexPosition().z() / CLHEP::cm << ")"
                                 << " parentid=" << aTrack->GetParentID();
#endif
  if (ekin > ekinMin_) {
    // Each track with energy above the threshold should be saved
    trkInfo_->putInHistory();
  }
}

void Phase2TrackingAction::PostUserTrackingAction(const G4Track* aTrack) {
  // Tracks in history may be upgraded to stored secondary tracks,
  // which cross the boundary between Tracker and Calo
  int id = aTrack->GetTrackID();
  bool ok = (trkInfo_->storeTrack() || currentTrack_->saved());
  if (trkInfo_->crossedBoundary()) {
    currentTrack_->setCrossedBoundaryPosMom(id, trkInfo_->getPositionAtBoundary(),
					    trkInfo_->getMomentumAtBoundary());
    ok = (ok || saveCaloBoundaryInformation_ || doFineCalo_);
  }
  if (ok) {
    currentTrack_->setToBeSaved();
  }

  bool withAncestor = (trkInfo_->getIDonCaloSurface() == id || trkInfo_->isAncestor());
  bool isInHistory = trkInfo_->isInHistory();

  trackManager_->addTrack(currentTrack_, aTrack, isInHistory, withAncestor);

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("Phase2TrackingAction")
      << "Phase2TrackingAction end track=" << id << "  "
      << aTrack->GetDefinition()->GetParticleName() << " proposed to be saved= " << ok
      << " end point " << aTrack->GetPosition();
#endif

  if (!isInHistory) {
    delete currentTrack_;
  }

  EndOfTrack et(aTrack);
  m_endOfTrackSignal(&et);
}
