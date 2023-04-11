#define EDM_ML_DEBUG

#include "SimG4Core/Notification/interface/NewTrackAction.h"
#include "SimG4Core/Notification/interface/TrackInformation.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4Track.hh"

NewTrackAction::NewTrackAction() {}

void NewTrackAction::primary(G4Track *aTrack) const { addUserInfoToPrimary(aTrack); }

void NewTrackAction::secondary(G4Track *aSecondary, const G4Track &mother, int flag) const {
  const TrackInformation *motherInfo = static_cast<const TrackInformation *>(mother.GetUserInformation());
  addUserInfoToSecondary(aSecondary, *motherInfo, flag);
  LogDebug("SimTrackManager") << "NewTrackAction: Add track " << aSecondary->GetTrackID() << " from mother "
                              << mother.GetTrackID();
}

void NewTrackAction::addUserInfoToPrimary(G4Track *aTrack) const {
  TrackInformation *trkInfo = new TrackInformation();
  trkInfo->setPrimary(true);
  trkInfo->setStoreTrack();
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
    trkInfo->setPrimary(true);
    trkInfo->setGenParticlePID(aTrack->GetDefinition()->GetPDGEncoding());
    trkInfo->setGenParticleP(aTrack->GetMomentum().mag());
  } else {
    trkInfo->setGenParticlePID(motherInfo.genParticlePID());
    trkInfo->setGenParticleP(motherInfo.genParticleP());
  }

  // Store if decay or conversion
  if (flag > 0) {
    trkInfo->setStoreTrack();
    trkInfo->putInHistory();
    trkInfo->setIDonCaloSurface(aTrack->GetTrackID(),
                                motherInfo.getIDCaloVolume(),
                                motherInfo.getIDLastVolume(),
                                aTrack->GetDefinition()->GetPDGEncoding(),
                                aTrack->GetMomentum().mag());
    LogDebug("SimG4CoreApplication") << "NewTrackAction: Id on calo surface " << trkInfo->getIDonCaloSurface()
                                     << " to be stored " << trkInfo->storeTrack();
  } else {
    // transfer calo ID from mother (to be checked in TrackingAction)
    trkInfo->setIDonCaloSurface(motherInfo.getIDonCaloSurface(),
                                motherInfo.getIDCaloVolume(),
                                motherInfo.getIDLastVolume(),
                                motherInfo.caloSurfaceParticlePID(),
                                motherInfo.caloSurfaceParticleP());
    LogDebug("SimG4CoreApplication") << "NewTrackAction: Id on calo surface " << trkInfo->getIDonCaloSurface();
  }

  if (motherInfo.hasCastorHit()) {
    trkInfo->setCastorHitPID(motherInfo.getCastorHitPID());
  }

  // manage ID of tracks in BTL to map them to SimTracks to be stored
  if (isInBTL(aTrack)) {
    if ((motherInfo.storeTrack() && motherInfo.isFromTtoBTL()) || motherInfo.isBTLdaughter()) {
      trkInfo->setBTLdaughter();
      trkInfo->setIdAtBTLentrance(motherInfo.idAtBTLentrance());
      LogDebug("SimG4CoreApplication") << "NewTrackAction: secondary in BTL " << trkInfo->isBTLdaughter()
                                       << " from mother ID " << trkInfo->idAtBTLentrance();
    }
  }

  aTrack->SetUserInformation(trkInfo);
}

bool NewTrackAction::isInBTL(const G4Track *aTrack) const {
  bool out = false;
  G4String tName(aTrack->GetVolume()->GetLogicalVolume()->GetRegion()->GetName());
  if ( tName == "FastTimerRegionBTL" || tName == "FastTimerRegionSensBTL" ) {
    out = true;
  }

  return out;
}
