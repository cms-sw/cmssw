#include "SimG4Core/SaveSimTrackAction/interface/SaveSimTrack.h"

#include "SimG4Core/Notification/interface/BeginOfTrack.h"
#include "SimG4Core/Notification/interface/TrackInformation.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4Track.hh"
#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"

SaveSimTrack::SaveSimTrack(edm::ParameterSet const & p) {

  pdgMin     = p.getUntrackedParameter<int>("MinimumPDGCode", 1000000);
  pdgMax     = p.getUntrackedParameter<int>("MaximumPDGCode", 2000000);

  edm::LogInfo("SaveSimTrack") << "SaveSimTrack:: Save Sim Track if PDG code "
			       << "lies between "  << pdgMin << " and " 
			       << pdgMax;
}

SaveSimTrack::~SaveSimTrack() {}
 
void SaveSimTrack::update(const BeginOfTrack * trk) {

  G4Track* theTrack = (G4Track*)((*trk)());
  TrackInformation * trkInfo = (TrackInformation *)(theTrack->GetUserInformation());
  if (trkInfo) {
    int pdg = std::abs(theTrack->GetDefinition()->GetPDGEncoding());
    if (pdg >= pdgMin && pdg <= pdgMax) {
      trkInfo->storeTrack(true);
      LogDebug("SaveSimTrack") << "Save SimTrack the Track " 
			       << theTrack->GetTrackID() << " Type " 
			       << theTrack->GetDefinition()->GetParticleName()
			       << " Momentum " << theTrack->GetMomentum()/MeV 
			       << " MeV/c";
    }
  }
}

