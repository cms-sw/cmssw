
#include "SimG4Core/CustomPhysics/interface/G4SQLoopProcess.h"
#include "G4SystemOfUnits.hh"
#include "G4Step.hh"
#include "G4ParticleDefinition.hh"
#include "G4VParticleChange.hh"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

G4SQLoopProcess::G4SQLoopProcess(const G4String& name, G4ProcessType type) : G4VContinuousProcess(name, type) {
  fParticleChange = new G4ParticleChange();
}

G4SQLoopProcess::~G4SQLoopProcess() { delete fParticleChange; }

G4VParticleChange* G4SQLoopProcess::AlongStepDoIt(const G4Track& track, const G4Step& step) {
  if (track.GetPosition() == posini)
    edm::LogInfo("G4SQLoopProcess::AlongStepDoIt")
        << "G4SQLoopProcess::AlongStepDoIt: G4SQLoopProcess::AlongStepDoIt  MomentumDirection "
        << track.GetMomentumDirection().eta() << " track GetPostion  " << track.GetPosition() / cm << " trackId "
        << track.GetTrackID() << " parentId: " << track.GetParentID() << " GlobalTime " << track.GetGlobalTime() / ns
        << " TotalEnergy: " << track.GetTotalEnergy() / GeV << " Velocity " << track.GetVelocity() / m / ns
        << std::endl;

  fParticleChange->Clear();
  fParticleChange->Initialize(track);
  fParticleChange->ProposeWeight(track.GetWeight());
  //Sbar not passing the following criteria are not of interest. They will not be reconstructable. A cut like this is required otherwise you will get Sbar infinitely looping.
  if (fabs(track.GetMomentumDirection().eta()) > 999. || fabs(track.GetPosition().z()) > 160 * centimeter) {
    edm::LogInfo("G4SQLoopProcess::AlongStepDoIt") << "Particle getting killed because too large z" << std::endl;
    fParticleChange->ProposeTrackStatus(fStopAndKill);
  }

  return fParticleChange;
}

G4double G4SQLoopProcess::AlongStepGetPhysicalInteractionLength(const G4Track& track,
                                                                G4double previousStepSize,
                                                                G4double currentMinimumStep,
                                                                G4double& proposedSafety,
                                                                G4GPILSelection* selection) {
  return 1. * centimeter;
}

G4double G4SQLoopProcess::GetContinuousStepLimit(const G4Track& track, G4double, G4double, G4double&) {
  return 1. * centimeter;  // seems irrelevant
}

void G4SQLoopProcess::StartTracking(G4Track* aTrack) { posini = aTrack->GetPosition(); }
