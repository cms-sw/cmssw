#include "SimG4Core/Notification/interface/G4TrackToParticleID.h"

#include "G4Track.hh"
#include "G4PrimaryParticle.hh"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

int G4TrackToParticleID::particleID(const G4Track* g4trk) {
  int pdg = g4trk->GetDefinition()->GetPDGEncoding();
  if (0 == pdg) {
    edm::LogWarning("SimG4CoreNotification")
        << "G4TrackToParticleID: unknown code 0 for trackId=" << g4trk->GetTrackID();
    pdg = -99;
  }
  return pdg;
}

int G4TrackToParticleID::particleID(const G4PrimaryParticle* ptr, const int id) {
  int pdg = ptr->GetPDGcode();
  if (0 == pdg) {
    edm::LogWarning("SimG4CoreNotification") << "G4TrackToParticleID: unknown code 0 for trackId=" << id;
    pdg = -99;
  }
  return pdg;
}

bool G4TrackToParticleID::isGammaElectronPositron(int pdgCode) {
  int pdg = std::abs(pdgCode);
  return (pdg == 11 || pdg == 22);
}

bool G4TrackToParticleID::isGammaElectronPositron(const G4Track* g4trk) {
  int pdg = std::abs(g4trk->GetDefinition()->GetPDGEncoding());
  return (pdg == 11 || pdg == 22);
}

bool G4TrackToParticleID::isMuon(int pdgCode) { return (std::abs(pdgCode) == 13); }

bool G4TrackToParticleID::isMuon(const G4Track* g4trk) {
  return (std::abs(g4trk->GetDefinition()->GetPDGEncoding()) == 13);
}

bool G4TrackToParticleID::isStableHadron(int pdgCode) {
  // pi+-, p, pbar, n, nbar, KL, K+-, light ions and anti-ions
  int pdg = std::abs(pdgCode);
  return (pdg == 211 || pdg == 2212 || pdg == 2112 || pdg == 130 || pdg == 321 || pdg == 1000010020 ||
          pdg == 1000010030 || pdg == 1000020030 || pdg == 1000020040);
}

bool G4TrackToParticleID::isStableHadronIon(const G4Track* g4trk) {
  // pi+-, p, pbar, n, nbar, KL, K+-, light ion and anti-ion, generic ion
  int pdg = std::abs(g4trk->GetDefinition()->GetPDGEncoding());
  return (pdg == 211 || pdg == 2212 || pdg == 2112 || pdg == 130 || pdg == 321 || pdg == 1000010020 ||
          pdg == 1000010030 || pdg == 1000020030 || pdg == 1000020040 || g4trk->GetDefinition()->IsGeneralIon());
}
