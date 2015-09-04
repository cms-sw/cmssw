#include <iostream>
#include "G4ParticleTable.hh"
#include "Randomize.hh"

#include "SimG4Core/CustomPhysics/interface/DummyChargeFlipProcess.h"

using namespace CLHEP;

DummyChargeFlipProcess::
DummyChargeFlipProcess(const G4String& processName) : 
      G4HadronicProcess(processName)
{
  AddDataSet(new G4HadronElasticDataSet);
}

DummyChargeFlipProcess::~DummyChargeFlipProcess()
{
}
 

void DummyChargeFlipProcess::
BuildPhysicsTable(const G4ParticleDefinition& aParticleType)
{
   if (!G4HadronicProcess::GetCrossSectionDataStore()) {
      return;
   }
   G4HadronicProcess::GetCrossSectionDataStore()->BuildPhysicsTable(aParticleType);
}


G4double DummyChargeFlipProcess::
GetMicroscopicCrossSection(
      const G4DynamicParticle* /*aParticle*/, const G4Element* element, G4double /*aTemp*/)
{

   return 30*millibarn*element->GetN(); 
}


G4bool
DummyChargeFlipProcess::
IsApplicable(const G4ParticleDefinition& aParticleType)
{
   if(aParticleType.GetParticleType()  == "rhadron")
    return true;
  else
    return false;
}


void 
DummyChargeFlipProcess::
DumpPhysicsTable(const G4ParticleDefinition& /*aParticleType*/)
{

}


G4VParticleChange *DummyChargeFlipProcess::PostStepDoIt(
  const G4Track &aTrack, const G4Step &/*aStep*/)
{
  G4ParticleChange * pc = new G4ParticleChange();
  pc->Initialize(aTrack);
  const G4DynamicParticle* aParticle = aTrack.GetDynamicParticle();
  //G4ParticleDefinition* aParticleDef = aParticle->GetDefinition();

  G4double   ParentEnergy  = aParticle->GetTotalEnergy();
  G4ThreeVector ParentDirection(aParticle->GetMomentumDirection());

  G4double energyDeposit = 0.0;
  G4double finalGlobalTime = aTrack.GetGlobalTime();

  G4int numberOfSecondaries = 1;
  pc->SetNumberOfSecondaries(numberOfSecondaries);
  const G4TouchableHandle thand = aTrack.GetTouchableHandle();

  // get current position of the track
  aTrack.GetPosition();
  // create a new track object
  G4ParticleTable* particleTable = G4ParticleTable::GetParticleTable();
  float randomParticle = G4UniformRand();
  G4ParticleDefinition * newType;
  if(randomParticle < 0.333)
    newType=particleTable->FindParticle(1009213);
  else if(randomParticle > 0.667)
    newType=particleTable->FindParticle(-1009213);
  else
    newType=particleTable->FindParticle(1009113);
     
  G4cout << "RHADRON: New charge = " << newType->GetPDGCharge() << G4endl;
       
  G4DynamicParticle * newP =  new G4DynamicParticle(newType,ParentDirection,ParentEnergy);
  G4Track* secondary = new G4Track( newP ,
				    finalGlobalTime ,
				    aTrack.GetPosition()
				    );
  // switch on good for tracking flag
  secondary->SetGoodForTrackingFlag();
  secondary->SetTouchableHandle(thand);
  // add the secondary track in the List
  pc->AddSecondary(secondary);

  // Kill the parent particle
  pc->ProposeTrackStatus( fStopAndKill ) ;
  pc->ProposeLocalEnergyDeposit(energyDeposit); 
  pc->ProposeGlobalTime( finalGlobalTime );
  // Clear NumberOfInteractionLengthLeft
  ClearNumberOfInteractionLengthLeft();

  return pc;
}

