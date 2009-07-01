#include "SimG4Core/GFlash/interface/GflashHadronShowerModel.h"
#include "SimG4Core/GFlash/interface/GflashHadronShowerProfile.h"
#include "SimG4Core/GFlash/interface/GflashNameSpace.h"
#include "SimG4Core/GFlash/interface/GflashHistogram.h"

#include "G4PionMinus.hh"
#include "G4PionPlus.hh"
#include "G4KaonMinus.hh"
#include "G4KaonPlus.hh"
#include "G4Proton.hh"
#include "G4AntiProton.hh"
#include "G4VProcess.hh"
#include "G4RegionStore.hh"
#include "G4FastSimulationManager.hh"

#include <vector>

#include "SimG4Core/GFlash/interface/GflashTrajectory.h"
#include "SimG4Core/GFlash/interface/GflashTrajectoryPoint.h"

GflashHadronShowerModel::GflashHadronShowerModel(G4String modelName, G4Region* envelope, edm::ParameterSet parSet)
  : G4VFastSimulationModel(modelName, envelope), theParSet(parSet)
{
  theProfile = new GflashHadronShowerProfile(envelope,parSet);
  theHisto = GflashHistogram::instance();
}

GflashHadronShowerModel::~GflashHadronShowerModel()
{
  if(theProfile) delete theProfile;
}

G4bool GflashHadronShowerModel::IsApplicable(const G4ParticleDefinition& particleType)
{
  return 
    &particleType == G4PionMinus::PionMinusDefinition() ||
    &particleType == G4PionPlus::PionPlusDefinition() ||
    &particleType == G4KaonMinus::KaonMinusDefinition() ||
    &particleType == G4KaonPlus::KaonPlusDefinition() ||
    //@@@turn-off AntiProton parameterization until it is completed
    //    &particleType == G4AntiProton::AntiProtonDefinition() ||
    &particleType == G4Proton::ProtonDefinition() ;
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
  if (fastTrack.GetPrimaryTrack()->GetKineticEnergy() < Gflash::energyCutOff) return trigger;

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

  theProfile->hadronicParameterization(fastTrack);

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
     (particleType == G4PionMinus::PionMinusDefinition() && procName == "WrappedPionMinusInelastic") ||
     (particleType == G4KaonPlus::KaonPlusDefinition() && procName == "WrappedKaonPlusInelastic") ||
     (particleType == G4KaonMinus::KaonMinusDefinition() && procName == "WrappedKaonMinusInelastic") ||
     //@@@turn-off AntiProton parameterization until it is completed
     //     (particleType == G4AntiProton::AntiProtonDefinition() && procName == "WrappedAntiProtonInelastic") ||
     (particleType == G4Proton::ProtonDefinition() && procName == "WrappedProtonInelastic")) {

    //skip to the second interaction if the first inelastic is a quasi-elastic like interaction
    //@@@ the cut may be optimized later

    const G4TrackVector* fSecondaryVector = fastTrack.GetPrimaryTrack()->GetStep()->GetSecondary();
    G4double leadingEnergy = 0.0;

    //loop over 'all' secondaries including those produced by continuous processes.
    //@@@may require an additional condition only for hadron interaction with the process name,
    //but it will not change the result anyway

    for (unsigned int isec = 0 ; isec < fSecondaryVector->size() ; isec++) {
      G4Track* fSecondaryTrack = (*fSecondaryVector)[isec];
      G4double secondaryEnergy = fSecondaryTrack->GetKineticEnergy();

      if(secondaryEnergy > leadingEnergy ) {
        leadingEnergy = secondaryEnergy;
      }
    }

    if((preStep->GetTotalEnergy()!=0) && 
       (leadingEnergy/preStep->GetTotalEnergy() < Gflash::QuasiElasticLike)) isFirst = true;

    //Fill debugging histograms and check information on secondaries -
    //remove after final implimentation

    if(theHisto->getStoreFlag()) {
      theHisto->preStepPosition->Fill(preStep->GetPosition().getRho()/cm);
      theHisto->postStepPosition->Fill(postStep->GetPosition().getRho()/cm);
      theHisto->deltaStep->Fill((postStep->GetPosition() - preStep->GetPosition()).getRho()/cm);
      theHisto->kineticEnergy->Fill(fastTrack.GetPrimaryTrack()->GetKineticEnergy()/GeV);
      theHisto->energyLoss->Fill(fabs(fastTrack.GetPrimaryTrack()->GetStep()->GetDeltaEnergy()/GeV));
      theHisto->energyRatio->Fill(leadingEnergy/preStep->GetTotalEnergy());
    }

 }
  return isFirst;
} 

G4bool GflashHadronShowerModel::excludeDetectorRegion(const G4FastTrack& fastTrack)
{
  G4bool isExcluded=false;
  
  //exclude regions where geometry are complicated 
  G4double eta =   fastTrack.GetPrimaryTrack()->GetPosition().pseudoRapidity() ;
  if(fabs(eta) > Gflash::EtaMax[Gflash::kESPM] && fabs(eta) < Gflash::EtaMin[Gflash::kENCA]) {
    //@@@remove this print statement later
    std::cout << "GflashHadronShowerModel: excluding region of eta = " << eta << std::endl;
    return true;  
  }
  else {
    G4StepPoint* postStep = fastTrack.GetPrimaryTrack()->GetStep()->GetPostStepPoint();

    Gflash::CalorimeterNumber kCalor = Gflash::getCalorimeterNumber(postStep->GetPosition()/cm);
    G4double distOut = 9999.0;
    //exclude the region where the shower starting point is outside parameterization envelopes
    if(kCalor==Gflash::kNULL) {
      isExcluded = true;
    }
    //@@@exclude the region where the shower starting point is too close to the end of
    //the hadronic envelopes (may need to be optimized further!)
    //@@@if we extend parameterization including Magnet/HO, we need to change this strategy
    else if(kCalor == Gflash::kHB) {
      distOut =  Gflash::Rmax[Gflash::kHB] - postStep->GetPosition().getRho()/cm;
      if (distOut < Gflash::MinDistanceToOut ) isExcluded = true;
    }
    else if(kCalor == Gflash::kHE) {
      distOut =  Gflash::Zmax[Gflash::kHE] - std::fabs(postStep->GetPosition().getZ()/cm);
      if (distOut < Gflash::MinDistanceToOut ) isExcluded = true;
    }
    //@@@remove this print statement later
    if(isExcluded) {
      std::cout << "GflashHadronShowerModel: skipping kCalor = " << kCalor << 
	" DistanceToOut " << distOut << std::endl;
    }
  }

  return isExcluded;
}
