//
// V.Ivanchenko 2013/10/19
// step limiter and killer for e+,e-
//
#include "SimG4Core/Application/interface/ElectronLimiter.h"

#include "G4ParticleDefinition.hh"
#include "G4VEnergyLossProcess.hh"
#include "G4LossTableManager.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4SystemOfUnits.hh"
#include "G4TransportationProcessType.hh"

ElectronLimiter::ElectronLimiter(const edm::ParameterSet & p)
 : G4VDiscreteProcess("eLimiter", fGeneral)
{
  // set Process Sub Type
  SetProcessSubType(static_cast<int>(STEP_LIMITER));

  minStepLimit = p.getParameter<double>("MinStepLimit")*mm;
  rangeCheckFlag = false;
  fieldCheckFlag = false;
  killTrack = false;

  fIonisation = 0;
  particle = 0;
}

ElectronLimiter::~ElectronLimiter() 
{}

void ElectronLimiter::BuildPhysicsTable(const G4ParticleDefinition& part)
{
  particle = &part;
  fIonisation = G4LossTableManager::Instance()->GetEnergyLossProcess(particle);
  /*
  std::cout << "ElectronLimiter::BuildPhysicsTable for " 
	    << particle->GetParticleName()
	    << " ioni: " << fIonisation << " rangeCheckFlag: " << rangeCheckFlag
	    << " fieldCheckFlag: " << fieldCheckFlag << std::endl;
  */
}

G4double 
ElectronLimiter::PostStepGetPhysicalInteractionLength(const G4Track& aTrack,
						      G4double, 
						      G4ForceCondition* condition)
{
  *condition = NotForced;
  
  G4double limit = DBL_MAX; 
  killTrack = false;

  if(rangeCheckFlag) {
    G4double safety = aTrack.GetStep()->GetPreStepPoint()->GetSafety();
    if(safety > minStepLimit) {
      G4double kinEnergy = aTrack.GetKineticEnergy();
      G4double range = fIonisation->GetRangeForLoss(kinEnergy,
						    aTrack.GetMaterialCutsCouple());
      if(safety >= range) { 
	killTrack = true; 
        limit = 0.0;
      }
    }
  }
  if(!killTrack && fieldCheckFlag) {
    limit = minStepLimit;
  }

  return limit;
}

inline G4VParticleChange* ElectronLimiter::PostStepDoIt(const G4Track& aTrack, 
							const G4Step&)
{
  fParticleChange.Initialize(aTrack);
  if(killTrack) {
    fParticleChange.ProposeTrackStatus(fStopAndKill);
    fParticleChange.ProposeLocalEnergyDeposit(aTrack.GetKineticEnergy());
    fParticleChange.SetProposedKineticEnergy(0.0);
  }
  return &fParticleChange;
}

inline G4double ElectronLimiter::GetMeanFreePath(const G4Track&,G4double,
						 G4ForceCondition*)
{
  return DBL_MAX;
}    
