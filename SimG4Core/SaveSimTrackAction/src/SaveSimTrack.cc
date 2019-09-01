#include "SimG4Core/SaveSimTrackAction/interface/SaveSimTrack.h"

#include "SimG4Core/Notification/interface/BeginOfTrack.h"
#include "SimG4Core/Notification/interface/TrackInformation.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"
#include "G4Track.hh"
#include <algorithm>

SaveSimTrack::SaveSimTrack(edm::ParameterSet const &p) {
  edm::ParameterSet ps = p.getParameter<edm::ParameterSet>("SaveSimTrack");
  pdgs_ = ps.getUntrackedParameter<std::vector<int>>("PDGCodes");

  edm::LogInfo("SaveSimTrack") << "SaveSimTrack:: Save Sim Track if PDG code "
                               << "is one from the list of " << pdgs_.size() << " items";
  for (unsigned int k = 0; k < pdgs_.size(); ++k)
    edm::LogInfo("SaveSimTrack") << "[" << k << "] " << pdgs_[k];
}

SaveSimTrack::~SaveSimTrack() {}

void SaveSimTrack::update(const BeginOfTrack *trk) {
  G4Track *theTrack = (G4Track *)((*trk)());
  TrackInformation *trkInfo = (TrackInformation *)(theTrack->GetUserInformation());
  if (trkInfo) {
    int pdg = theTrack->GetDefinition()->GetPDGEncoding();
    if (std::find(pdgs_.begin(), pdgs_.end(), pdg) != pdgs_.end()) {
      trkInfo->storeTrack(true);
      LogDebug("SaveSimTrack") << "Save SimTrack the Track " << theTrack->GetTrackID() << " Type "
                               << theTrack->GetDefinition()->GetParticleName() << " Momentum "
                               << theTrack->GetMomentum() / MeV << " MeV/c";
    }
  }
}
