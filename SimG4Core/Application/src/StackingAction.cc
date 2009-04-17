#include "SimG4Core/Application/interface/StackingAction.h"
#include "SimG4Core/Notification/interface/CurrentG4Track.h"
#include "SimG4Core/Notification/interface/NewTrackAction.h"
#include "SimG4Core/Notification/interface/TrackInformation.h"
#include "SimG4Core/Notification/interface/TrackInformationExtractor.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4VProcess.hh"
#include "G4PhysicalVolumeStore.hh"
 
StackingAction::StackingAction(const edm::ParameterSet & p): tracker(0),
							     calo(0), muon(0) {
  trackNeutrino  = p.getParameter<bool>("TrackNeutrino");
  killHeavy      = p.getParameter<bool>("KillHeavy");
  kmaxIon        = p.getParameter<double>("IonThreshold")*MeV;
  kmaxProton     = p.getParameter<double>("ProtonThreshold")*MeV;
  kmaxNeutron    = p.getParameter<double>("NeutronThreshold")*MeV;
  savePDandCinTracker = p.getUntrackedParameter<bool>("SavePrimaryDecayProductsAndConversionsInTracker",false);
  savePDandCinCalo    = p.getUntrackedParameter<bool>("SavePrimaryDecayProductsAndConversionsInCalo",false);
  savePDandCinMuon    = p.getUntrackedParameter<bool>("SavePrimaryDecayProductsAndConversionsInMuon",false);
  saveFirstSecondary  = p.getUntrackedParameter<bool>("SaveFirstLevelSecondary",false);

  edm::LogInfo("SimG4CoreApplication") << "StackingAction initiated with"
				       << " flag for saving decay products in "
				       << " Tracker: " << savePDandCinTracker
                                       << " in Calo: " << savePDandCinCalo
                                       << " in Muon: " << savePDandCinMuon
				       << "\n               saveFirstSecondary"
				       << ": " << saveFirstSecondary
				       << " Flag for tracking neutrino: "
				       << trackNeutrino << " Killing Flag "
				       << killHeavy << " protons below " 
				       << kmaxProton <<" MeV, neutrons below "
				       << kmaxNeutron << " MeV and ions"
				       << " below " << kmaxIon << " MeV\n";

  initPointer();
}

StackingAction::~StackingAction() {}

G4ClassificationOfNewTrack StackingAction::ClassifyNewTrack(const G4Track * aTrack) {

  // G4 interface part
  G4ClassificationOfNewTrack classification = fUrgent;
  int flag = 0;

  NewTrackAction newTA;
  if (aTrack->GetCreatorProcess()==0 || aTrack->GetParentID()==0)
    newTA.primary(aTrack);
  else {
    const G4Track * mother = CurrentG4Track::track();
    if ((savePDandCinTracker && isThisVolume(aTrack->GetTouchable(),tracker))||
	(savePDandCinCalo && isThisVolume(aTrack->GetTouchable(),calo)) ||
	(savePDandCinMuon && isThisVolume(aTrack->GetTouchable(),muon)))
      flag = isItPrimaryDecayProductOrConversion(aTrack, *mother);
    if (saveFirstSecondary) flag = isItFromPrimary(*mother, flag);
    newTA.secondary(aTrack, *mother, flag);
    if (killHeavy) {
      int    pdg = aTrack->GetDefinition()->GetPDGEncoding();
      double ke  = aTrack->GetKineticEnergy()/MeV;
      if (((pdg/1000000000 == 1) && (((pdg/10000)%100) > 0) && 
	   (((pdg/10)%100) > 0) && (ke<kmaxIon)) || 
	  ((pdg == 2212) && (ke < kmaxProton)) ||
	  ((pdg == 2112) && (ke < kmaxNeutron))) classification = fKill;
    }
    if (!trackNeutrino) {
      int    pdg = std::abs(aTrack->GetDefinition()->GetPDGEncoding());
      if (pdg == 12 || pdg == 14 || pdg == 16 || pdg == 18) 
	classification = fKill;
    }
    LogDebug("SimG4CoreApplication") << "StackingAction:Classify Track "
				     << aTrack->GetTrackID() << " Parent " 
				     << aTrack->GetParentID() << " Type "
				     << aTrack->GetDefinition()->GetParticleName() 
				     << " K.E. " << aTrack->GetKineticEnergy()/MeV
				     << " MeV as " << classification 
				     << " Flag " << flag;
  }
  return classification;
}

void StackingAction::NewStage() {}

void StackingAction::PrepareNewEvent() {}

void StackingAction::initPointer() {

  const G4PhysicalVolumeStore * pvs = G4PhysicalVolumeStore::GetInstance();
  if (pvs) {
    std::vector<G4VPhysicalVolume*>::const_iterator pvcite;
    for (pvcite = pvs->begin(); pvcite != pvs->end(); pvcite++) {
      if (savePDandCinTracker) {
        if ((*pvcite)->GetName() == "Tracker") tracker = (*pvcite);
      }
      if (savePDandCinCalo) {
        if ((*pvcite)->GetName() == "CALO")    calo    = (*pvcite);
      }
      if (savePDandCinMuon) {
        if ((*pvcite)->GetName() == "MUON")    muon    = (*pvcite);
      }
      if ( (!savePDandCinTracker || tracker) && (!savePDandCinCalo || calo) &&
	   (!savePDandCinMuon || muon ) ) break;
    }
    edm::LogInfo("SimG4CoreApplication") << "Pointers for Tracker " << tracker
                                         << ", Calo " << calo << ", Muon "
					 << muon;
    if (tracker) edm::LogInfo("SimG4CoreApplication") << "Tracker vol name "
						      << tracker->GetName();
    if (calo)    edm::LogInfo("SimG4CoreApplication")<< "Calorimeter vol name "
						     << calo->GetName();
    if (muon)    edm::LogInfo("SimG4CoreApplication") << "Muon vol name "
						      << muon->GetName();
  }
}

bool StackingAction::isThisVolume(const G4VTouchable* touch, 
				  G4VPhysicalVolume* pv) const {

  int level = ((touch->GetHistoryDepth())+1);
  if (level >= 3) {
    int  ii   = level - 3;
    bool flag = (touch->GetVolume(ii) == pv);
    return flag;
  }
  return false;
}

int StackingAction::isItPrimaryDecayProductOrConversion(const G4Track * aTrack,
							const G4Track & mother) const {

  int flag = 0;
  TrackInformationExtractor extractor;
  const TrackInformation & motherInfo(extractor(mother));
  // Check whether mother is a primary
  if (motherInfo.isPrimary()) {
    if (aTrack->GetCreatorProcess()->GetProcessType() == fDecay &&
	aTrack->GetCreatorProcess()->GetProcessName() == "Decay") flag = 1;
    else if (aTrack->GetCreatorProcess()->GetProcessType() == fElectromagnetic &&
	     aTrack->GetCreatorProcess()->GetProcessName() == "conv") flag = 2;
  }
  return flag;
}

int StackingAction::isItFromPrimary(const G4Track & mother, int flagIn) const {

  int flag = flagIn;
  if (flag != 1) {
    TrackInformationExtractor extractor;
    const TrackInformation & motherInfo(extractor(mother));
    if (motherInfo.isPrimary()) flag = 3;
  }
  return flag;
}
