#include "SimG4Core/KillSecondaries/interface/KillSecondariesTrackAction.h"

#include "SimG4Core/Notification/interface/BeginOfTrack.h"
#include "SimG4Core/Notification/interface/TrackInformation.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4Track.hh"

using namespace CLHEP;

KillSecondariesTrackAction::KillSecondariesTrackAction(edm::ParameterSet const &p) {
  killHeavy = p.getParameter<bool>("KillHeavy");
  kmaxIon = p.getParameter<double>("IonThreshold") * MeV;
  kmaxProton = p.getParameter<double>("ProtonThreshold") * MeV;
  kmaxNeutron = p.getParameter<double>("NeutronThreshold") * MeV;

  edm::LogInfo("KillSecondaries") << "KillSecondariesTrackAction:: Killing"
                                  << " Flag " << killHeavy << " protons below " << kmaxProton << " MeV, neutrons below "
                                  << kmaxNeutron << " MeV and ions below " << kmaxIon << " MeV\n";
}

KillSecondariesTrackAction::~KillSecondariesTrackAction() {}

void KillSecondariesTrackAction::update(const BeginOfTrack *trk) {
  if (killHeavy) {
    G4Track *theTrack = (G4Track *)((*trk)());
    TrackInformation *trkInfo = (TrackInformation *)(theTrack->GetUserInformation());
    if (trkInfo) {
      int pdg = theTrack->GetDefinition()->GetPDGEncoding();
      if (!(trkInfo->isPrimary())) {  // Only secondary particles
        double ke = theTrack->GetKineticEnergy() / MeV;
        if ((((pdg / 1000000000 == 1 && ((pdg / 10000) % 100) > 0 && ((pdg / 10) % 100) > 0)) && (ke < kmaxIon)) ||
            ((pdg == 2212) && (ke < kmaxProton)) || ((pdg == 2112) && (ke < kmaxNeutron))) {
          theTrack->SetTrackStatus(fStopAndKill);
          edm::LogInfo("KillSecondaries")
              << "Kill Track " << theTrack->GetTrackID() << " Type " << theTrack->GetDefinition()->GetParticleName()
              << " Kinetic Energy " << ke << " MeV";
        }
      }
    }
  }
}
