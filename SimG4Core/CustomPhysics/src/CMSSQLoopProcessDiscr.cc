
#include "SimG4Core/CustomPhysics/interface/CMSSQLoopProcessDiscr.h"
#include "G4SystemOfUnits.hh"
#include "G4Step.hh"
#include "G4ParticleDefinition.hh"
#include "G4VParticleChange.hh"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

CMSSQLoopProcessDiscr::CMSSQLoopProcessDiscr(double mass, const G4String& name, G4ProcessType type)
    : G4VDiscreteProcess(name, type) {
  fParticleChange = new G4ParticleChange();
  fParticleChange->ClearDebugFlag();
  GenMass = mass;
}

CMSSQLoopProcessDiscr::~CMSSQLoopProcessDiscr() { delete fParticleChange; }

G4VParticleChange* CMSSQLoopProcessDiscr::PostStepDoIt(const G4Track& track, const G4Step& step) {
  G4Track* mytr = const_cast<G4Track*>(&track);
  mytr->SetPosition(posini);
  if (mytr->GetGlobalTime() / ns > 4990)
    edm::LogWarning("CMSSQLoopProcess::AlongStepDoIt")
        << "going to loose the particle because the GlobalTime is getting close to 5000" << std::endl;

  fParticleChange->Clear();
  fParticleChange->Initialize(track);

  //adding secondary antiS
  fParticleChange->SetNumberOfSecondaries(1);
  G4DynamicParticle* replacementParticle =
      new G4DynamicParticle(CMSAntiSQ::AntiSQ(GenMass), track.GetMomentumDirection(), track.GetKineticEnergy());
  fParticleChange->AddSecondary(replacementParticle, globaltimeini);

  //killing original AntiS
  fParticleChange->ProposeTrackStatus(fStopAndKill);

  // note SL: this way of working makes a very long history of the track,
  // which all get saved recursively in SimTracks. If the cross section
  // is too low such that 10's of thousands of iterations are needed, then
  // this becomes too heavy to swallow writing out this history.
  // So if we ever need very small cross sections, then we really need
  // to change this way of working such that we can throw away all original
  // tracks and only save the one that interacted.

  return fParticleChange;
}

G4double CMSSQLoopProcessDiscr::PostStepGetPhysicalInteractionLength(const G4Track& track,
                                                                     G4double previousStepSize,
                                                                     G4ForceCondition* condition) {
  *condition = NotForced;
  G4double intLength =
      DBL_MAX;  //by default the interaction length is super large, only when outside tracker make it 0 to be sure it will do the reset to the original position
  G4Track* mytr = const_cast<G4Track*>(&track);
  if (sqrt(pow(mytr->GetPosition().rho(), 2)) >
      2.45 *
          centimeter) {  //this is an important cut for the looping: if the radius of the particle is largher than 2.45cm its interaction length becomes 0 which means it will get killed
    // updated from 2.5 to 2.45 so that the Sbar does not start to hit the support of the new inner tracker which was added in 2018
    intLength = 0.0;  //0 interaction length means particle will directly interact.
  }
  return intLength;
}

G4double CMSSQLoopProcessDiscr::GetMeanFreePath(const G4Track&, G4double, G4ForceCondition*) { return DBL_MAX; }

void CMSSQLoopProcessDiscr::StartTracking(G4Track* aTrack) {
  posini = aTrack->GetPosition();
  globaltimeini = aTrack->GetGlobalTime();
}
