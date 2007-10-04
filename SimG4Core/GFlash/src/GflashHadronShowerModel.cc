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


GflashHadronShowerModel::GflashHadronShowerModel(G4String modelName, G4Region* envelope)
  : G4VFastSimulationModel(modelName, envelope), 
    theProfile(new GflashHadronShowerProfile(envelope)),
    theHitMaker(new GFlashHitMaker())
{
  std::cout << " GflashHadronicShowerModel Name " << modelName << std::endl; 

  //temporary book-keeping histograms 
  f = new TFile("Gflash_histo.root","recreate");
  f->cd();

  h_enormal  = new TH1F("h_enormal","Kinetic Energy Tested in Normal GPIL",200,0.0,200.);
  h_ewrapper = new TH1F("h_ewrapper","Kinetic Energy Tested after Wrapper GPIL",200,0.0,200.);

  h_energy = new TH1F("h_energy","Kinetic Energy",200,0.0,200.);
  h_esum   = new TH1F("h_esum","Parameterized Energy",200,0.0,200.);
  h_elos   = new TH1F("h_elos","Energy Loss in this step",200,0.0,200.);
  h_dout   = new TH1F("h_dout","Distance to Out",200,0.0,200.);
  h_ssp    = new TH1F("h_ssp","Shower Starting Point",200,100.0,300.);
  h_ratio  = new TH1F("h_ratio","Energy Loss/Kinetic",100,0.0,2.0);

}

GflashHadronShowerModel::~GflashHadronShowerModel()
{
  delete theProfile;
  delete theHitMaker;

  std::cout << "Saving Gflash histograms to Gflash_histo.root" <<std::endl;
  f->Write();
  f->Close();

}

G4bool GflashHadronShowerModel::IsApplicable(const G4ParticleDefinition& particleType)
{
  return 
    &particleType == G4PionMinus::PionMinusDefinition() ||
    &particleType == G4PionPlus::PionPlusDefinition();
}

G4bool GflashHadronShowerModel::ModelTrigger(const G4FastTrack& fastTrack)
{
  // ModelTrigger returns false for the CMS Hadronic Shower Model if it
  // is not tested from the G4 wrapper process, DelayStepingProcess for 
  // G4HadronInelasticProcess. As a temporary implementation, the track
  // status is set to fPostponeToNextEvent a the wrapper process before
  // if ModelTrigger is called through PostStepGPIL for the process type
  // fParameterisation. The better implmentation may be using via 
  // G4VUserTrackInformation of each track. 

  G4bool trigger = false;

  // mininum energy cutoff to parameterize
  if (fastTrack.GetPrimaryTrack()->GetKineticEnergy() < 1.0*GeV) return trigger;

  // check whether this is called from the normal GPIL or the wrapper process GPIL
  if(fastTrack.GetPrimaryTrack()->GetTrackStatus() == fPostponeToNextEvent) {
    
    // Shower pameterization start at the first inelastic interaction point
    G4bool isInelastic  = isFirstInelasticInteraction(fastTrack);
    
    // Other conditions
    if(isInelastic) {
      trigger = (!excludeDetectorRegion(fastTrack));
    }
    h_ewrapper->Fill(fastTrack.GetPrimaryTrack()->GetKineticEnergy()/GeV);
  }
  else {
    h_enormal->Fill(fastTrack.GetPrimaryTrack()->GetKineticEnergy()/GeV);
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

  h_energy->Fill(fastTrack.GetPrimaryTrack()->GetKineticEnergy()/GeV);
  h_esum->Fill(energySum);

}

G4bool GflashHadronShowerModel::isFirstInelasticInteraction(const G4FastTrack& fastTrack)
{
  G4bool isFirst=false;

  G4StepPoint* postStep = fastTrack.GetPrimaryTrack()->GetStep()->GetPostStepPoint();
  G4String procName = postStep->GetProcessDefinedStep()->GetProcessName();  
  G4ParticleDefinition* particleType = fastTrack.GetPrimaryTrack()->GetDefinition();

  //@@@ this part is still temporary and the cut for the variable ratio should be optimized later

  if((particleType == G4PionPlus::PionPlusDefinition() && procName == "WrappedPionPlusInelastic") || 
     (particleType == G4PionMinus::PionMinusDefinition() && procName == "WrappedPionMinusInelastic")) {

    h_ssp->Fill(fastTrack.GetPrimaryTrack()->GetStep()->GetPreStepPoint()->GetPosition().getRho()/cm);
    h_elos->Fill(fabs(fastTrack.GetPrimaryTrack()->GetStep()->GetDeltaEnergy()/GeV));

    G4double ratio = 0.0;
    G4double energy = fastTrack.GetPrimaryTrack()->GetKineticEnergy();
    if (energy > 0) {
      ratio = fabs(fastTrack.GetPrimaryTrack()->GetStep()->GetDeltaEnergy()/energy);
      h_ratio->Fill(ratio);  
    }
    if(ratio > 0.1) isFirst=true;
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

  GflashCalorimeterNumber kColor = theProfile->getCalorimeterNumber(fastTrack);

  //@@@ need a proper scale
  const G4double minDistantToOut = 10.;
  
  if(kColor == kHB || kColor == kHE) {
    G4double distOut = fastTrack.GetEnvelopeSolid()->
      DistanceToOut(fastTrack.GetPrimaryTrackLocalPosition(),
		    fastTrack.GetPrimaryTrackLocalDirection());

    h_dout->Fill(distOut/cm);

    if (distOut < minDistantToOut ) {
      std::cout << "excluding region for dsitOut = " << distOut << std::endl;
      isExcluded = true;
    }
  }

  return isExcluded;
}
