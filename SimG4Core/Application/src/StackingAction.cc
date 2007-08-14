#include "SimG4Core/Application/interface/StackingAction.h"
#include "SimG4Core/Notification/interface/CurrentG4Track.h"
#include "SimG4Core/Notification/interface/NewTrackAction.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
 
StackingAction::StackingAction(const edm::ParameterSet & p) {
  savePrimaryDecayProductsAndConversions = p.getUntrackedParameter<bool>("SavePrimaryDecayProductsAndConversions",false);
  suppressHeavy = p.getUntrackedParameter<bool>("SuppressHeavy", false);
  pmaxIon       = p.getUntrackedParameter<double>("IonThreshold", 50.0)*MeV;
  pmaxProton    = p.getUntrackedParameter<double>("ProtonThreshold", 50.0)*MeV;
  pmaxNeutron   = p.getUntrackedParameter<double>("NeutronThreshold", 50.0)*MeV;
  edm::LogInfo("SimG4CoreApplication") << "StackingAction initiated with"
				       << " flag for saving decay products "
				       <<savePrimaryDecayProductsAndConversions
				       << " Suppression Flag " << suppressHeavy
				       << " protons below " << pmaxProton 
				       << " MeV/c, neutrons below "
				       << pmaxNeutron << " MeV/c and ions"
				       << " below " << pmaxIon << " MeV/c\n";

}

StackingAction::~StackingAction() {}

G4ClassificationOfNewTrack StackingAction::ClassifyNewTrack(const G4Track * aTrack) {

  // G4 interface part
  G4ClassificationOfNewTrack classification = fUrgent;

  NewTrackAction newTA(savePrimaryDecayProductsAndConversions);
  if (aTrack->GetCreatorProcess()==0 || aTrack->GetParentID()==0)
    newTA.primary(aTrack);
  else {
    const G4Track * mother = CurrentG4Track::track();
    newTA.secondary(aTrack, *mother);
    if (suppressHeavy) {
      int    pdg = aTrack->GetDefinition()->GetPDGEncoding();
      double pp  = aTrack->GetMomentum().mag()/MeV;
      if (((pdg/1000000000 == 1) && (((pdg/10000)%100) > 0) && 
	   (((pdg/10)%100) > 0) && (pp<pmaxIon)) || 
	  ((pdg == 2212) && (pp < pmaxProton)) ||
	  ((pdg == 2112) && (pp < pmaxNeutron))) classification = fKill;
    }
    LogDebug("SimG4CoreApplication") << "StackingAction:Classify Track "
				     << aTrack->GetTrackID() << " Parent " 
				     << aTrack->GetParentID() << " Type "
				     << aTrack->GetDefinition()->GetParticleName() 
				     << " Momentum " << aTrack->GetMomentum().mag()/MeV
				     << " MeV/c as " << classification;
  }
  return classification;
}

void StackingAction::NewStage() {}

void StackingAction::PrepareNewEvent() {}


