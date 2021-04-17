#include "SimG4Core/CustomPhysics/interface/DummyChargeFlipProcess.h"

#include <iostream>
#include "G4ParticleTable.hh"
#include "Randomize.hh"
#include "G4NeutronElasticXS.hh"
#include "G4Step.hh"
#include "G4TrackStatus.hh"
#include "G4Element.hh"

using namespace CLHEP;

DummyChargeFlipProcess::DummyChargeFlipProcess(const G4String& pname) : G4HadronicProcess(pname, fHadronic) {
  AddDataSet(new G4NeutronElasticXS());
  fPartChange = new G4ParticleChange();
}

DummyChargeFlipProcess::~DummyChargeFlipProcess() { delete fPartChange; }

G4bool DummyChargeFlipProcess::IsApplicable(const G4ParticleDefinition& aParticleType) {
  return (aParticleType.GetParticleType() == "rhadron");
}

G4VParticleChange* DummyChargeFlipProcess::PostStepDoIt(const G4Track& aTrack, const G4Step&) {
  fPartChange->Initialize(aTrack);
  const G4DynamicParticle* aParticle = aTrack.GetDynamicParticle();

  G4double ParentEnergy = aParticle->GetTotalEnergy();
  const G4ThreeVector& ParentDirection(aParticle->GetMomentumDirection());

  G4double energyDeposit = 0.0;
  G4double finalGlobalTime = aTrack.GetGlobalTime();

  G4int numberOfSecondaries = 1;
  fPartChange->SetNumberOfSecondaries(numberOfSecondaries);
  const G4TouchableHandle& thand = aTrack.GetTouchableHandle();

  // get current position of the track
  aTrack.GetPosition();
  // create a new track object
  G4ParticleTable* particleTable = G4ParticleTable::GetParticleTable();
  float randomParticle = G4UniformRand();
  G4ParticleDefinition* newType;
  if (randomParticle < 0.333)
    newType = particleTable->FindParticle(1009213);
  else if (randomParticle > 0.667)
    newType = particleTable->FindParticle(-1009213);
  else
    newType = particleTable->FindParticle(1009113);

  //G4cout << "RHADRON: New charge = " << newType->GetPDGCharge() << G4endl;

  G4DynamicParticle* newP = new G4DynamicParticle(newType, ParentDirection, ParentEnergy);
  G4Track* secondary = new G4Track(newP, finalGlobalTime, aTrack.GetPosition());
  // switch on good for tracking flag
  secondary->SetGoodForTrackingFlag();
  secondary->SetTouchableHandle(thand);
  // add the secondary track in the List
  fPartChange->AddSecondary(secondary);

  // Kill the parent particle
  fPartChange->ProposeTrackStatus(fStopAndKill);
  fPartChange->ProposeLocalEnergyDeposit(energyDeposit);
  fPartChange->ProposeGlobalTime(finalGlobalTime);

  ClearNumberOfInteractionLengthLeft();

  return fPartChange;
}
