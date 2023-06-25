
#include "SimG4Core/Notification/interface/MCTruthUtil.h"
#include "SimG4Core/Notification/interface/TrackInformation.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "G4Track.hh"

void MCTruthUtil::primary(G4Track *aTrack) {
  TrackInformation *trkInfo = new TrackInformation();
  trkInfo->setPrimary(true);
  trkInfo->setStoreTrack();
  trkInfo->setGenParticlePID(aTrack->GetDefinition()->GetPDGEncoding());
  trkInfo->setGenParticleP(aTrack->GetMomentum().mag());
  trkInfo->setMCTruthID(aTrack->GetTrackID());
  aTrack->SetUserInformation(trkInfo);
}

void MCTruthUtil::secondary(G4Track *aTrack, const G4Track &mother, int flag) {
  auto motherInfo = static_cast<const TrackInformation *>(mother.GetUserInformation());
  auto trkInfo = new TrackInformation();
  LogDebug("SimG4CoreApplication") << "MCTruthUtil called for " << aTrack->GetTrackID() << " mother "
                                   << motherInfo->isPrimary() << " flag " << flag;

  // Take care of cascade decays
  if (flag == 1) {
    trkInfo->setPrimary(true);
    trkInfo->setStoreTrack();
    trkInfo->setGenParticlePID(aTrack->GetDefinition()->GetPDGEncoding());
    trkInfo->setGenParticleP(aTrack->GetMomentum().mag());
    trkInfo->setMCTruthID(aTrack->GetTrackID());
  } else {
    // secondary
    trkInfo->setGenParticlePID(motherInfo->genParticlePID());
    trkInfo->setGenParticleP(motherInfo->genParticleP());
    trkInfo->setMCTruthID(motherInfo->mcTruthID());
  }

  // Store if decay or conversion
  if (flag > 0) {
    trkInfo->setStoreTrack();
    trkInfo->setIDonCaloSurface(aTrack->GetTrackID(),
                                motherInfo->getIDCaloVolume(),
                                motherInfo->getIDLastVolume(),
                                aTrack->GetDefinition()->GetPDGEncoding(),
                                aTrack->GetMomentum().mag());
  } else {
    // transfer calo ID from mother (to be checked in TrackingAction)
    trkInfo->setIDonCaloSurface(motherInfo->getIDonCaloSurface(),
                                motherInfo->getIDCaloVolume(),
                                motherInfo->getIDLastVolume(),
                                motherInfo->caloSurfaceParticlePID(),
                                motherInfo->caloSurfaceParticleP());
  }

  // for Run1 and Run2
  if (motherInfo->hasCastorHit()) {
    trkInfo->setCastorHitPID(motherInfo->getCastorHitPID());
  }

  // manage ID of tracks in BTL to map them to SimTracks to be stored
  if (isInBTL(aTrack)) {
    if ((motherInfo->storeTrack() && motherInfo->isFromTtoBTL()) || motherInfo->isBTLdaughter()) {
      trkInfo->setBTLdaughter();
      trkInfo->setIdAtBTLentrance(motherInfo->idAtBTLentrance());
      LogDebug("SimG4CoreApplication") << "NewTrackAction: secondary in BTL " << trkInfo->isBTLdaughter()
                                       << " from mother ID " << trkInfo->idAtBTLentrance();
    }
  }

  aTrack->SetUserInformation(trkInfo);
}

bool MCTruthUtil::isInBTL(const G4Track *aTrack) {
  bool out = false;
  G4String tName(aTrack->GetVolume()->GetLogicalVolume()->GetRegion()->GetName());
  if (tName == "FastTimerRegionBTL" || tName == "FastTimerRegionSensBTL") {
    out = true;
  }
  return out;
}
