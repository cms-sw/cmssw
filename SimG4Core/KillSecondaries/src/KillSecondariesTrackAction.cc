#include "SimG4Core/KillSecondaries/interface/KillSecondariesTrackAction.h"

#include "SimG4Core/Notification/interface/BeginOfTrack.h"
#include "SimG4Core/Notification/interface/TrackInformation.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4Track.hh"

KillSecondariesTrackAction::KillSecondariesTrackAction(edm::ParameterSet const & p) {

  suppress    = p.getUntrackedParameter<bool>("SuppressHeavy", false);
  pmaxIon     = p.getUntrackedParameter<double>("IonThreshold", 50.0)*MeV;
  pmaxProton  = p.getUntrackedParameter<double>("ProtonThreshold", 50.0)*MeV;
  pmaxNeutron = p.getUntrackedParameter<double>("NeutronThreshold", 50.0)*MeV;

  std::cout << "KillSecondariesTrackAction:: Suppression Flag " << suppress
	    << " protons below " << pmaxProton << " MeV/c, neutrons below "
	    << pmaxNeutron << " and ions below " << pmaxIon << " MeV/c\n";
}

KillSecondariesTrackAction::~KillSecondariesTrackAction() {}
 
void KillSecondariesTrackAction::update(const BeginOfTrack * trk) {

  if (suppress) {
    G4Track* theTrack = (G4Track*)((*trk)());
    TrackInformation * trkInfo = (TrackInformation *)(theTrack->GetUserInformation());
    if (trkInfo) {
      int pdg = theTrack->GetDefinition()->GetPDGEncoding();
      if (!(trkInfo->isPrimary())) { // Only secondary particles
	double pp = theTrack->GetMomentum().mag()/MeV;
	if (((pdg/1000000000 == 1 && ((pdg/10000)%100) > 0 &&
	      ((pdg/10)%100) > 0)) && (pp<pmaxIon)) {
	  theTrack->SetTrackStatus(fStopAndKill);
	  edm::LogInfo("KillSecondaries") << "Kill Track " << theTrack->GetTrackID()
					  << " Type " << theTrack->GetDefinition()->GetParticleName()
					  << " Momentum " << pp << " MeV/c";
	}
	if ((pdg == 2212) && (pp < pmaxProton)) {
	  theTrack->SetTrackStatus(fStopAndKill);
	  edm::LogInfo("KillSecondaries") << "Kill Track " << theTrack->GetTrackID()
					  << " Type " << theTrack->GetDefinition()->GetParticleName()
					  << " Momentum " << pp << " MeV/c";
      	}
	if ((pdg == 2112) && (pp < pmaxNeutron)) {
	  theTrack->SetTrackStatus(fStopAndKill);
	  edm::LogInfo("KillSecondaries") << "Kill Track " << theTrack->GetTrackID()
					  << " Type " << theTrack->GetDefinition()->GetParticleName()
					  << " Momentum " << pp << " MeV/c";
	}
      }
    }
  }
}

