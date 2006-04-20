#include "SimG4Core/CustomPhysics/interface/DummyChargeFlipProcess.h"
#include "G4ParticleTable.hh"
#include "CLHEP/Random/RandFlat.h"

#include <iostream>

using namespace std;

DummyChargeFlipProcess::DummyChargeFlipProcess(const std::string & processName) : 
      G4HadronicProcess(processName)
{  AddDataSet(new G4HadronElasticDataSet); }

DummyChargeFlipProcess::~DummyChargeFlipProcess() {}
 

void DummyChargeFlipProcess::BuildPhysicsTable(const G4ParticleDefinition & aParticleType)
{
    if (!G4HadronicProcess::GetCrossSectionDataStore()) return;
    G4HadronicProcess::GetCrossSectionDataStore()->BuildPhysicsTable(aParticleType);
}

G4double DummyChargeFlipProcess::GetMicroscopicCrossSection(const G4DynamicParticle * aParticle, 
							    const G4Element* anElement, double aTemp)
{
    // gives the microscopic cross section in GEANT4 internal units
    if (!G4HadronicProcess::GetCrossSectionDataStore()) 
    {
	G4Exception("DummyChargeFlipProcess", "007", FatalException,
		    "no cross section data Store");
	return DBL_MIN;
    }
    return 30*millibarn*anElement->GetN(); 
}

bool DummyChargeFlipProcess::IsApplicable(const G4ParticleDefinition& aParticleType)
{
    if (aParticleType.GetParticleType() == "rhadron")
	return true;
    else
	return false;
}

void DummyChargeFlipProcess::DumpPhysicsTable(const G4ParticleDefinition & aParticleType)
{
   if (!G4HadronicProcess::GetCrossSectionDataStore()) 
   {
       G4Exception("DummyChargeFlipProcess", "111", JustWarning, 
		   "DummyChargeFlipProcess: no cross section data set");
       return;
   }
   G4HadronicProcess::GetCrossSectionDataStore()->DumpPhysicsTable(aParticleType);
}


G4VParticleChange * DummyChargeFlipProcess::PostStepDoIt(const G4Track &aTrack, 
							 const G4Step &aStep)
{
    cout << "Sign flip ....!!"<< endl;
    SetDispatch(this);
    G4ParticleChange * pc = new G4ParticleChange();
    pc->Initialize(aTrack);
    const G4DynamicParticle * aParticle = aTrack.GetDynamicParticle();
    G4ParticleDefinition * aParticleDef = aParticle->GetDefinition();

    //Bug!!! Use kinetic energy
    //  G4double   ParentEnergy  = aParticle->GetTotalEnergy();
    double   ParentEnergy  = aParticle->GetKineticEnergy();
    G4ThreeVector ParentDirection(aParticle->GetMomentumDirection());

    double energyDeposit = 0.0;
    double finalGlobalTime = aTrack.GetGlobalTime();

    int numberOfSecondaries = 1;
    pc->SetNumberOfSecondaries(numberOfSecondaries);
    const G4TouchableHandle thand = aTrack.GetTouchableHandle();

    // get current position of the track
    aTrack.GetPosition();
    // create a new track object
    G4ParticleTable * particleTable = G4ParticleTable::GetParticleTable();
    float randomParticle = RandFlat::shoot();
    G4ParticleDefinition * newType = aParticleDef;
    if(randomParticle < 0.333)
        newType=particleTable->FindParticle(1009213);
    else if(randomParticle > 0.667)
        newType=particleTable->FindParticle(-1009213);
    else
	newType=particleTable->FindParticle(1009113);
     
    cout << "RHADRON: New charge = " << newType->GetPDGCharge() << endl;
       
    G4DynamicParticle * newP =  new G4DynamicParticle(newType,ParentDirection,ParentEnergy);
    G4Track* secondary = new G4Track(newP,finalGlobalTime,aTrack.GetPosition());
    // switch on good for tracking flag
    secondary->SetGoodForTrackingFlag();
    secondary->SetTouchableHandle(thand);
    // add the secondary track in the List
    pc->AddSecondary(secondary);

    // Kill the parent particle
    pc->ProposeTrackStatus(fStopAndKill) ;
    pc->ProposeLocalEnergyDeposit(energyDeposit); 
    pc->ProposeGlobalTime(finalGlobalTime);
    // Clear NumberOfInteractionLengthLeft
    ClearNumberOfInteractionLengthLeft();

    return pc;
}

