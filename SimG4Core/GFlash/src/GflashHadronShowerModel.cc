#include "SimG4Core/GFlash/interface/GflashHadronShowerModel.h"
#include "SimG4Core/GFlash/interface/GflashHadronShowerProfile.h"
#include "SimG4Core/G4gflash/src/GFlashHitMaker.hh"

#include "GFlashEnergySpot.hh"
#include "G4PionMinus.hh"
#include "G4PionPlus.hh"
#include "G4TransportationManager.hh"
#include "G4VSensitiveDetector.hh"
#include "G4TouchableHandle.hh"
#include "G4VProcess.hh"

#include "Randomize.hh"
#include "CLHEP/GenericFunctions/IncompleteGamma.hh"
#include <vector>

G4StepPoint* tmpStepPoint = 0;

GflashHadronShowerModel::GflashHadronShowerModel(G4String modelName, G4Region* envelope)
  : G4VFastSimulationModel(modelName, envelope), 
    theProfile(new GflashHadronShowerProfile(envelope)),
    theHitMaker(new GFlashHitMaker())
{
  G4cout << " GflashHadronicShowerModel Name " << modelName << G4endl; 
}

GflashHadronShowerModel::~GflashHadronShowerModel()
{
  delete theProfile;
  delete theHitMaker;
}

G4bool GflashHadronShowerModel::IsApplicable(const G4ParticleDefinition& particleType)
{
  return 
    &particleType == G4PionMinus::PionMinusDefinition() ||
    &particleType == G4PionPlus::PionPlusDefinition();
}

G4bool GflashHadronShowerModel::ModelTrigger(const G4FastTrack& fastTrack)
{
  G4bool trigger = false;

  // Trigger parameterisation only above 1 GeV
  if (fastTrack.GetPrimaryTrack()->GetKineticEnergy() < 1.0*GeV) return trigger;

  // Shower pameterization start at the first inelastic interaction point
  G4bool isInelastic  = isFirstInelasticInteraction(fastTrack);

  // Other conditions
  if(isInelastic) {
    trigger = (!excludeDetectorRegion(fastTrack));
  }

  return trigger;
}

void GflashHadronShowerModel::DoIt(const G4FastTrack& fastTrack, G4FastStep& fastStep)
{
  // Kill the parameterised particle:

  fastStep.KillPrimaryTrack();
  fastStep.ProposePrimaryTrackPathLength(0.0);
  fastStep.ProposeTotalEnergyDeposited(fastTrack.GetPrimaryTrack()->GetKineticEnergy());

  // Parameterize shower shape and get resultant energy spots
  theProfile->hadronicParameterization(fastTrack);
  std::vector<GFlashEnergySpot> energySpotList = theProfile->getEnergySpotList();

  // Make hits
  G4double energySum = 0.0;
  for (size_t i = 0 ; i < energySpotList.size() ; i++) {
    theHitMaker->make(&energySpotList[i], &fastTrack);
    energySum += energySpotList[i].GetEnergy();
  }
}

G4bool GflashHadronShowerModel::isFirstInelasticInteraction(const G4FastTrack& fastTrack)
{
  G4bool isFirst=false;

  G4StepPoint* postStep = fastTrack.GetPrimaryTrack()->GetStep()->GetPostStepPoint();
  G4String procName = postStep->GetProcessDefinedStep()->GetProcessName();  
  G4ParticleDefinition* particleType = fastTrack.GetPrimaryTrack()->GetDefinition();

  //@@@ this part is temporary and needs to be correctly implemented later
  if((particleType == G4PionPlus::PionPlusDefinition() && procName == "PionPlusInelastic") || 
     (particleType == G4PionMinus::PionMinusDefinition() && procName == "PionMinusInelastic")) {
    isFirst=true;
  }
  return isFirst;
} 

G4bool GflashHadronShowerModel::excludeDetectorRegion(const G4FastTrack& fastTrack)
{
  G4bool isExcluded=false;
  
  //exclude regions where geometry are complicated 
  G4double eta =   fastTrack.GetPrimaryTrack()->GetMomentum().pseudoRapidity() ;
  if(fabs(eta) > 1.30 && fabs(eta) < 1.57) return true;  

  //exclude the region where the shower starting point is too close to the end of 
  //the hadronic envelopes (may need to be optimized further!)

  GflashCalorimeterNumber kColor = theProfile->getCalorimeterNumber(fastTrack);

  //@@@ need a proper scale
  const G4double minDistantToOut = 10.;
  
  if(kColor == kHB || kColor == kHE) {
    G4double distOut = fastTrack.GetEnvelopeSolid()->
      DistanceToOut(fastTrack.GetPrimaryTrackLocalPosition(),
		    fastTrack.GetPrimaryTrackLocalDirection());
    if (distOut < minDistantToOut ) isExcluded = true;
  }

  return isExcluded;
}
