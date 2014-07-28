#include "SimG4Core/Notification/interface/G4TrackToParticleID.h"

#include "G4Track.hh"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

int G4TrackToParticleID::particleID(const G4Track * g4trk)
{
    int particleID_ = g4trk->GetDefinition()->GetPDGEncoding();
    if (0 == particleID_) {
      edm::LogWarning("SimG4CoreNotification") 
	<< "G4TrackToParticleID: unknown code for track Id = " << g4trk->GetTrackID();
      particleID_ = -99;
    }
    return particleID_;
}
