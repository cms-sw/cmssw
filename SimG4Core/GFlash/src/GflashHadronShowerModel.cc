#include "SimG4Core/GFlash/interface/GflashHadronShowerModel.h"
#include "SimG4Core/GFlash/interface/GflashHadronShowerProfile.h"
#include "SimG4Core/GFlash/interface/GflashEnergySpot.h"
#include "SimG4Core/GFlash/interface/GflashHistogram.h"

#include "G4VSensitiveDetector.hh"
#include "G4VPhysicalVolume.hh"

#include "G4PionMinus.hh"
#include "G4PionPlus.hh"
#include "G4TransportationManager.hh"
#include "G4TouchableHandle.hh"
#include "G4VProcess.hh"
#include "G4RegionStore.hh"
#include "G4FastSimulationManager.hh"

#include <vector>

#include "SimG4Core/GFlash/interface/GflashTrajectory.h"
#include "SimG4Core/GFlash/interface/GflashTrajectoryPoint.h"

GflashHadronShowerModel::GflashHadronShowerModel(G4String modelName, G4Region* envelope)
  : G4VFastSimulationModel(modelName, envelope)
{
  theProfile = new GflashHadronShowerProfile(envelope);
  theHisto = GflashHistogram::instance();

  theGflashStep = new G4Step();
  theGflashNavigator = new G4Navigator();
  theGflashTouchableHandle = new G4TouchableHistory();

  theGflashNavigator->SetWorldVolume(G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking()->GetWorldVolume());

}

GflashHadronShowerModel::~GflashHadronShowerModel()
{
  if(theProfile) delete theProfile;
  if(theGflashStep) delete theGflashStep;
}

G4bool GflashHadronShowerModel::IsApplicable(const G4ParticleDefinition& particleType)
{
  return 
    &particleType == G4PionMinus::PionMinusDefinition() ||
    &particleType == G4PionPlus::PionPlusDefinition();
}

G4bool GflashHadronShowerModel::ModelTrigger(const G4FastTrack& fastTrack)
{
  // ModelTrigger returns false for Gflash Hadronic Shower Model if it is not
  // tested from the corresponding wrapper process, GflashHadronWrapperProcess. 
  // Temporarily the track status is set to fPostponeToNextEvent at the wrapper
  // process before ModelTrigger is really tested for the second time through 
  // PostStepGPIL of enviaged parameterization processes.  The better 
  // implmentation may be using via G4VUserTrackInformation of each track, which
  // requires to modify a geant source code of stepping (G4SteppingManager2)

  G4bool trigger = false;

  // mininum energy cutoff to parameterize
  if (fastTrack.GetPrimaryTrack()->GetKineticEnergy() < 1.0*GeV) return trigger;

  // check whether this is called from the normal GPIL or the wrapper process GPIL
  if(fastTrack.GetPrimaryTrack()->GetTrackStatus() == fPostponeToNextEvent ) {

    // Shower pameterization start at the first inelastic interaction point
    G4bool isInelastic  = isFirstInelasticInteraction(fastTrack);
    
    // Other conditions
    if(isInelastic) {
      trigger = (!excludeDetectorRegion(fastTrack));
    }
  }

  return trigger;

}

void GflashHadronShowerModel::DoIt(const G4FastTrack& fastTrack, G4FastStep& fastStep)
{
  // Kill the parameterised particle:

  fastStep.ProposeTotalEnergyDeposited(fastTrack.GetPrimaryTrack()->GetKineticEnergy());

  // Parameterize shower shape and get resultant energy spots
  theProfile->clearSpotList();
  theProfile->hadronicParameterization(fastTrack);

  std::vector<GflashEnergySpot>& energySpotList = theProfile->getEnergySpotList();

  // Make hits
  G4double timeGlobal = fastTrack.GetPrimaryTrack()->GetStep()->GetPreStepPoint()->GetGlobalTime();
  
  std::vector<GflashEnergySpot>::const_iterator spotIter    = energySpotList.begin();
  std::vector<GflashEnergySpot>::const_iterator spotIterEnd = energySpotList.end();
  
   for( ; spotIter != spotIterEnd; spotIter++){

    // to make a different time for each fake step. (+1.0 is arbitrary)
    timeGlobal += 0.0001*nanosecond;

    // fill equivalent changes to a (fake) step associated with a spot 

    theGflashStep->SetTrack(const_cast<G4Track*>(fastTrack.GetPrimaryTrack()));
    theGflashStep->GetPostStepPoint()->SetGlobalTime(timeGlobal);
    theGflashStep->GetPreStepPoint()->SetPosition(spotIter->getPosition());
    theGflashStep->GetPostStepPoint()->SetPosition(spotIter->getPosition());
    theGflashStep->GetPostStepPoint()->SetProcessDefinedStep(const_cast<G4VProcess*> (fastTrack.GetPrimaryTrack()->GetStep()->GetPostStepPoint()->GetProcessDefinedStep()));

    //put touchable for each energy spot
    theGflashNavigator->LocateGlobalPointAndUpdateTouchable(spotIter->getPosition(),theGflashTouchableHandle(), false);
    theGflashStep->GetPreStepPoint()->SetTouchableHandle(theGflashTouchableHandle);
    theGflashStep->SetTotalEnergyDeposit(spotIter->getEnergy());
    
    // Send G4Step information to Hit/Dig if the volume is sensitive
    // Copied from G4SteppingManager.cc

    G4VPhysicalVolume* aCurrentVolume = theGflashStep->GetPreStepPoint()->GetPhysicalVolume();

    if( aCurrentVolume != 0 ) {
      theGflashStep->GetPreStepPoint()->SetSensitiveDetector(aCurrentVolume->GetLogicalVolume()->GetSensitiveDetector());
      G4VSensitiveDetector* aSensitive = theGflashStep->GetPreStepPoint()->GetSensitiveDetector();
      
      if( aSensitive != 0 ) {
	aSensitive->Hit(theGflashStep);
      }
    }
  }

  fastStep.KillPrimaryTrack();
  fastStep.ProposePrimaryTrackPathLength(0.0);

}

G4bool GflashHadronShowerModel::isFirstInelasticInteraction(const G4FastTrack& fastTrack)
{
  G4bool isFirst=false;

  G4StepPoint* preStep = fastTrack.GetPrimaryTrack()->GetStep()->GetPreStepPoint();
  G4StepPoint* postStep = fastTrack.GetPrimaryTrack()->GetStep()->GetPostStepPoint();

  G4String procName = postStep->GetProcessDefinedStep()->GetProcessName();  
  G4ParticleDefinition* particleType = fastTrack.GetPrimaryTrack()->GetDefinition();

  //@@@ this part is still temporary and the cut for the variable ratio should be optimized later

  if((particleType == G4PionPlus::PionPlusDefinition() && procName == "WrappedPionPlusInelastic") || 
     (particleType == G4PionMinus::PionMinusDefinition() && procName == "WrappedPionMinusInelastic")) {

    G4double energy = fastTrack.GetPrimaryTrack()->GetKineticEnergy();

    G4double ratio = 0.0;
    if (energy > 0) {
      ratio = fabs(fastTrack.GetPrimaryTrack()->GetStep()->GetDeltaEnergy()/energy);
    }

    if(ratio > 0.1) isFirst=true;

    //Fill debugging histograms and check information on secondaries -
    //remove after final implimentation

    if(theHisto->getStoreFlag()) {
      theHisto->preStepPosition->Fill(preStep->GetPosition().getRho()/cm);
      theHisto->postStepPosition->Fill(postStep->GetPosition().getRho()/cm);
      theHisto->deltaStep->Fill((postStep->GetPosition() - preStep->GetPosition()).getRho()/cm);
      theHisto->kineticEnergy->Fill(energy);
      theHisto->energyLoss->Fill(fabs(fastTrack.GetPrimaryTrack()->GetStep()->GetDeltaEnergy()/GeV));
    }

 }
  return isFirst;
} 

G4bool GflashHadronShowerModel::excludeDetectorRegion(const G4FastTrack& fastTrack)
{
  G4bool isExcluded=false;
  
  //exclude regions where geometry are complicated 
  G4double eta =   fastTrack.GetPrimaryTrack()->GetMomentum().pseudoRapidity() ;
  if(fabs(eta) > 1.30 && fabs(eta) < 1.57) {
    std::cout << "excluding region of eta = " << eta << std::endl;
    return true;  
  }
  //exclude the region where the shower starting point is too close to the end of 
  //the hadronic envelopes (may need to be optimized further!)

  /*
  Gflash::CalorimeterNumber kColor = theProfile->getCalorimeterNumber(fastTrack);

  //@@@ need a proper scale
  const G4double minDistantToOut = 10.;
  

  if(kColor == kHB || kColor == kHE) {
    G4double distOut = fastTrack.GetEnvelopeSolid()->
      DistanceToOut(fastTrack.GetPrimaryTrackLocalPosition(),
		    fastTrack.GetPrimaryTrackLocalDirection());

    if (distOut < minDistantToOut ) {
      std::cout << "excluding region for dsitOut = " << distOut << std::endl;
      isExcluded = true;
    }
  }
  */

  return isExcluded;
}
