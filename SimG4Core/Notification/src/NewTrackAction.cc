
#include "SimG4Core/Notification/interface/NewTrackAction.h"
#include "SimG4Core/Notification/interface/TrackInformation.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4Track.hh"

NewTrackAction::NewTrackAction() {}

void NewTrackAction::primary(const G4Track *aTrack) const { primary(const_cast<G4Track *>(aTrack)); }

void NewTrackAction::primary(G4Track *aTrack) const { addUserInfoToPrimary(aTrack); }

void NewTrackAction::secondary(const G4Track *aSecondary, const G4Track &mother, int flag) const {
  secondary(const_cast<G4Track *>(aSecondary), mother, flag);
}

void NewTrackAction::secondary(G4Track *aSecondary, const G4Track &mother, int flag) const {
  const TrackInformation *motherInfo = static_cast<const TrackInformation *>(mother.GetUserInformation());
  addUserInfoToSecondary(aSecondary, *motherInfo, flag);
  LogDebug("SimTrackManager") << "NewTrackAction: Add track " << aSecondary->GetTrackID() << " from mother "
                              << mother.GetTrackID();
}

void NewTrackAction::addUserInfoToPrimary(G4Track *aTrack) const {
  TrackInformation *trkInfo = new TrackInformation();
  trkInfo->isPrimary(true);
  trkInfo->storeTrack(true);
  trkInfo->putInHistory();
  trkInfo->setGenParticlePID(aTrack->GetDefinition()->GetPDGEncoding());
  trkInfo->setGenParticleP(aTrack->GetMomentum().mag());
  aTrack->SetUserInformation(trkInfo);
}

void NewTrackAction::addUserInfoToSecondary(G4Track *aTrack, const TrackInformation &motherInfo, int flag) const {
  TrackInformation *trkInfo = new TrackInformation();
  LogDebug("SimG4CoreApplication") << "NewTrackAction called for " << aTrack->GetTrackID() << " mother "
                                   << motherInfo.isPrimary() << " flag " << flag;

  // Take care of cascade decays
  if (flag == 1) {
    trkInfo->isPrimary(true);
    trkInfo->setGenParticlePID(aTrack->GetDefinition()->GetPDGEncoding());
    trkInfo->setGenParticleP(aTrack->GetMomentum().mag());
  } else {
    trkInfo->setGenParticlePID(motherInfo.genParticlePID());
    trkInfo->setGenParticleP(motherInfo.genParticleP());
  }

  // Store if decay or conversion
  if (flag > 0) {
    trkInfo->storeTrack(true);
    trkInfo->putInHistory();
    trkInfo->setIDonCaloSurface(aTrack->GetTrackID(),
                                motherInfo.getIDCaloVolume(),
                                motherInfo.getIDLastVolume(),
                                aTrack->GetDefinition()->GetPDGEncoding(),
                                aTrack->GetMomentum().mag());
  } else {
    // transfer calo ID from mother (to be checked in TrackingAction)
    trkInfo->setIDonCaloSurface(motherInfo.getIDonCaloSurface(),
                                motherInfo.getIDCaloVolume(),
                                motherInfo.getIDLastVolume(),
                                motherInfo.caloSurfaceParticlePID(),
                                motherInfo.caloSurfaceParticleP());
  }

  if (motherInfo.hasCastorHit()) {
    trkInfo->setCastorHitPID(motherInfo.getCastorHitPID());
  }

  aTrack->SetUserInformation(trkInfo);
}
