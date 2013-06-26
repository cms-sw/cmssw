#include "SimG4Core/Notification/interface/G4TrackToParticleID.h"

#include "G4Track.hh"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//#define DebugLog

G4TrackToParticleID::G4TrackToParticleID() {}

int G4TrackToParticleID::particleID(const G4Track * g4trk)
{
    int particleID_ = g4trk->GetDefinition()->GetPDGEncoding();
#ifdef DebugLog
    if ( particleID_ > 1000000000 ) {
      LogDebug("SimG4CoreNotification") << "G4TrackToParticleID ion code = " << particleID_ ;
    }
#endif
    if (particleID_ != 0) return particleID_;
    edm::LogWarning("SimG4CoreNotification") << "G4TrackToParticleID: unknown code for track Id = " << g4trk->GetTrackID();
    return -99;
}

G4TrackToParticleID::~G4TrackToParticleID() {}
