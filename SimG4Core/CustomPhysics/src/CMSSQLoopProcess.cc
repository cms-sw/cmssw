
#include "SimG4Core/CustomPhysics/interface/CMSSQLoopProcess.h"
#include "G4SystemOfUnits.hh"
#include "G4Step.hh"
#include "G4ParticleDefinition.hh"
#include "G4VParticleChange.hh"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

CMSSQLoopProcess::CMSSQLoopProcess(const G4String& name, G4ProcessType type) : G4VContinuousProcess(name, type) {
  fParticleChange = new G4ParticleChange();
}

CMSSQLoopProcess::~CMSSQLoopProcess() { delete fParticleChange; }

G4VParticleChange* CMSSQLoopProcess::AlongStepDoIt(const G4Track& track, const G4Step& step) {
  if (track.GetPosition() == posini)
    edm::LogInfo("CMSSQLoopProcess::AlongStepDoIt")
        << "CMSSQLoopProcess::AlongStepDoIt: CMSSQLoopProcess::AlongStepDoIt  MomentumDirection "
        << track.GetMomentumDirection().eta() << " track GetPostion  " << track.GetPosition() / cm << " trackId "
        << track.GetTrackID() << " parentId: " << track.GetParentID() << " GlobalTime " << track.GetGlobalTime() / ns
        << " TotalEnergy: " << track.GetTotalEnergy() / GeV << " Velocity " << track.GetVelocity() / m / ns
        << std::endl;

  fParticleChange->Clear();
  fParticleChange->Initialize(track);
  fParticleChange->ProposeWeight(track.GetWeight());
  //Sbar not passing the following criteria are not of interest. They will not be reconstructable. A cut like this is required otherwise you will get Sbar infinitely looping.
  if (fabs(track.GetMomentumDirection().eta()) > 999. || fabs(track.GetPosition().z()) > 160 * centimeter) {
    edm::LogInfo("CMSSQLoopProcess::AlongStepDoIt") << "Particle getting killed because too large z" << std::endl;
    fParticleChange->ProposeTrackStatus(fStopAndKill);
  }

  return fParticleChange;
}

G4double CMSSQLoopProcess::AlongStepGetPhysicalInteractionLength(const G4Track& track,
                                                                 G4double previousStepSize,
                                                                 G4double currentMinimumStep,
                                                                 G4double& proposedSafety,
                                                                 G4GPILSelection* selection) {
  return 1. * centimeter;
}

G4double CMSSQLoopProcess::GetContinuousStepLimit(const G4Track& track, G4double, G4double, G4double&) {
  return 1. * centimeter;  // seems irrelevant
}

void CMSSQLoopProcess::StartTracking(G4Track* aTrack) { posini = aTrack->GetPosition(); }
