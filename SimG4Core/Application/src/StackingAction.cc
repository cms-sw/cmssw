#include "SimG4Core/Application/interface/StackingAction.h"
#include "SimG4Core/Notification/interface/CurrentG4Track.h"
#include "SimG4Core/Notification/interface/NewTrackAction.h"
#include "SimG4Core/Notification/interface/TrackInformation.h"
#include "SimG4Core/Notification/interface/TrackInformationExtractor.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4VProcess.hh"
#include "G4EmProcessSubType.hh"
#include "G4LogicalVolumeStore.hh"
#include "G4RegionStore.hh"
#include "Randomize.hh"

#include<algorithm>

//#define DebugLog

StackingAction::StackingAction(const edm::ParameterSet & p) {

  trackNeutrino  = p.getParameter<bool>("TrackNeutrino");
  killHeavy      = p.getParameter<bool>("KillHeavy");
  kmaxIon        = p.getParameter<double>("IonThreshold")*MeV;
  kmaxProton     = p.getParameter<double>("ProtonThreshold")*MeV;
  kmaxNeutron    = p.getParameter<double>("NeutronThreshold")*MeV;
  killDeltaRay   = p.getParameter<bool>("KillDeltaRay");
  maxTrackTime   = p.getParameter<double>("MaxTrackTime")*ns;
  maxTrackTimes  = p.getParameter<std::vector<double> >("MaxTrackTimes");
  maxTimeNames   = p.getParameter<std::vector<std::string> >("MaxTimeNames");
  savePDandCinTracker = p.getUntrackedParameter<bool>("SavePrimaryDecayProductsAndConversionsInTracker",false);
  savePDandCinCalo    = p.getUntrackedParameter<bool>("SavePrimaryDecayProductsAndConversionsInCalo",false);
  savePDandCinMuon    = p.getUntrackedParameter<bool>("SavePrimaryDecayProductsAndConversionsInMuon",false);
  saveFirstSecondary  = p.getUntrackedParameter<bool>("SaveFirstLevelSecondary",false);
  killInCalo = false;
  killInCaloEfH = false;

  // Russian Roulette
  nRusRoEcal = p.getParameter<double>("RusRoEcalNeutron");
  nRusRoHcal = p.getParameter<double>("RusRoHcalNeutron");
  nRusRoEcalLim = p.getParameter<double>("RusRoEcalNeutronLimit");
  nRusRoHcalLim = p.getParameter<double>("RusRoHcalNeutronLimit");
  pRusRoEcal = p.getParameter<double>("RusRoEcalProton");
  pRusRoHcal = p.getParameter<double>("RusRoHcalProton");
  pRusRoEcalLim = p.getParameter<double>("RusRoEcalProtonLimit");
  pRusRoHcalLim = p.getParameter<double>("RusRoHcalProtonLimit");
  regionEcal = 0;
  regionHcal = 0;

  if ( p.exists("TestKillingOptions") ) {

    killInCalo = (p.getParameter<edm::ParameterSet>("TestKillingOptions")).getParameter<bool>("KillInCalo");
    killInCaloEfH = (p.getParameter<edm::ParameterSet>("TestKillingOptions")).getParameter<bool>("KillInCaloEfH");
    edm::LogWarning("SimG4CoreApplication") << " *** Activating special test killing options in StackingAction \n"
                                            << " *** Kill secondaries in Calorimetetrs volume = " << killInCalo << "\n"
                                            << " *** Kill electromagnetic secondaries from hadrons in Calorimeters volume = " << killInCaloEfH;

  }

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
				       << " below " << kmaxIon << " MeV\n"
				       << "               kill tracks with "
				       << "time larger than " << maxTrackTime
				       << " ns and kill Delta Ray flag set to "
				       << killDeltaRay;
  for (unsigned int i=0; i<maxTrackTimes.size(); i++) {
    maxTrackTimes[i] *= ns;
    edm::LogInfo("SimG4CoreApplication") << "StackingAction::MaxTrackTime for "
					 << maxTimeNames[i] << " is " 
					 << maxTrackTimes[i];
  }
  initPointer();
}

StackingAction::~StackingAction() {}

G4ClassificationOfNewTrack StackingAction::ClassifyNewTrack(const G4Track * aTrack) {

  // G4 interface part
  G4ClassificationOfNewTrack classification = fUrgent;
  int flag = 0;

  NewTrackAction newTA;
  if (aTrack->GetCreatorProcess()==0 || aTrack->GetParentID()==0) {
    /*
    std::cout << "StackingAction: primary weight= " 
	      << aTrack->GetWeight() << " "
	      << aTrack->GetDefinition()->GetParticleName()
	      << " " << aTrack->GetKineticEnergy()
	      << " Id=" << aTrack->GetTrackID()
	      << "  trackInfo " << aTrack->GetUserInformation() 
	      << std::endl;
    */
    newTA.primary(aTrack);
    if (!trackNeutrino) {
      int    pdg = std::abs(aTrack->GetDefinition()->GetPDGEncoding());
      if (pdg == 12 || pdg == 14 || pdg == 16 || pdg == 18) {
	classification = fKill;
      }
    }
  } else if (aTrack->GetTouchable() == 0) {
    edm::LogError("SimG4CoreApplication")
      << "StackingAction: no touchable for track " << aTrack->GetTrackID()
      << " from " << aTrack->GetParentID()
      << " with PDG code " << aTrack->GetDefinition()->GetParticleName();
    classification = fKill;
  } else {
    const G4Track * mother = CurrentG4Track::track();
    if ((savePDandCinTracker && isThisVolume(aTrack->GetTouchable(),tracker))||
	(savePDandCinCalo && isThisVolume(aTrack->GetTouchable(),calo)) ||
	(savePDandCinMuon && isThisVolume(aTrack->GetTouchable(),muon)))
      flag = isItPrimaryDecayProductOrConversion(aTrack, *mother);
    if (saveFirstSecondary) flag = isItFromPrimary(*mother, flag);
    newTA.secondary(aTrack, *mother, flag);

    if (aTrack->GetTrackStatus() == fStopAndKill) classification = fKill;
    if (killHeavy && classification != fKill) {
      int    pdg = aTrack->GetDefinition()->GetPDGEncoding();
      double ke  = aTrack->GetKineticEnergy()/MeV;
      if (((pdg/1000000000 == 1) && (((pdg/10000)%100) > 0) && 
	   (((pdg/10)%100) > 0) && (ke<kmaxIon)) || 
	  ((pdg == 2212) && (ke < kmaxProton)) ||
	  ((pdg == 2112) && (ke < kmaxNeutron))) classification = fKill;
    }
    if (!trackNeutrino  && classification != fKill) {
      int    pdg = std::abs(aTrack->GetDefinition()->GetPDGEncoding());
      if (pdg == 12 || pdg == 14 || pdg == 16 || pdg == 18) 
	classification = fKill;
    }
    if (isItLongLived(aTrack)) classification = fKill;
    if (killDeltaRay) {
      if (aTrack->GetCreatorProcess()->GetProcessType() == fElectromagnetic &&
	  aTrack->GetCreatorProcess()->GetProcessSubType() == fIonisation)
	classification = fKill;
    }
    if (killInCalo && classification != fKill) {
      if ( isThisVolume(aTrack->GetTouchable(),calo)) { 
        classification = fKill; 
      }
    }
    if (killInCaloEfH && classification != fKill) {
      int    pdg = aTrack->GetDefinition()->GetPDGEncoding();
      int    pdgMother = mother->GetDefinition()->GetPDGEncoding();
      if ( (pdg == 22 || std::abs(pdg) == 11) && (std::abs(pdgMother) < 11 || std::abs(pdgMother) > 17) && pdgMother != 22  ) {
        if ( isThisVolume(aTrack->GetTouchable(),calo)) { 
          classification = fKill; 
        }
      }
    }
    if((regionEcal || regionHcal) && classification != fKill) {
      int    pdg = aTrack->GetDefinition()->GetPDGEncoding();
      if(2112 == pdg || 2212 == pdg) {
	G4Region* reg = aTrack->GetVolume()->GetLogicalVolume()->GetRegion();
        double prob = 1.0;
        double elim = 0.0;
	if(reg == regionEcal) {
          if(2112 == pdg) {
	    prob = nRusRoEcal;
            elim = nRusRoEcalLim;
	  } else {
	    prob = pRusRoEcal;
            elim = pRusRoEcalLim;
	  }
	} else if(reg == regionHcal) {
          if(2112 == pdg) {
	    prob = nRusRoHcal;
            elim = nRusRoHcalLim;
	  } else {
	    prob = pRusRoHcal;
            elim = pRusRoHcalLim;
	  }
	}
        if(prob < 1.0 && aTrack->GetKineticEnergy() < elim) {
          if(G4UniformRand() < prob) {
            const_cast<G4Track*>(aTrack)->SetWeight(aTrack->GetWeight()/prob);
          } else {
	    classification = fKill;
          }
	}
      }  
    }
  double wt2 = aTrack->GetWeight();
  /*
  if(wt2 != 1.0) { 
    G4Region* reg = aTrack->GetVolume()->GetLogicalVolume()->GetRegion();
    std::cout << "StackingAction: weight= " << wt2 << " "
	      << aTrack->GetDefinition()->GetParticleName()
	      << " " << aTrack->GetKineticEnergy()
	      << " Id=" << aTrack->GetTrackID()
	      << " IdP=" << aTrack->GetParentID();
    const G4VProcess* pr = aTrack->GetCreatorProcess();
    if(pr) std::cout << " from  " << pr->GetProcessName();
    if(reg) std::cout << " in  " << reg->GetName()
		      << "  trackInfo " << aTrack->GetUserInformation(); 
    std::cout << std::endl;
  }
  */
#ifdef DebugLog
    LogDebug("SimG4CoreApplication") << "StackingAction:Classify Track "
				     << aTrack->GetTrackID() << " Parent " 
				     << aTrack->GetParentID() << " Type "
				     << aTrack->GetDefinition()->GetParticleName() 
				     << " K.E. " << aTrack->GetKineticEnergy()/MeV
				     << " MeV from process/subprocess " 
				     << aTrack->GetCreatorProcess()->GetProcessType() << "|"
				     << aTrack->GetCreatorProcess()->GetProcessSubType()
				     << " as " << classification << " Flag " << flag;
#endif
  }
  return classification;
}

void StackingAction::NewStage() {}

void StackingAction::PrepareNewEvent() {}

void StackingAction::initPointer() {

  const G4LogicalVolumeStore * lvs = G4LogicalVolumeStore::GetInstance();
  if (lvs) {
    std::vector<G4LogicalVolume*>::const_iterator lvcite;
    for (lvcite = lvs->begin(); lvcite != lvs->end(); lvcite++) {
      if (savePDandCinTracker) {
        if (strcmp("Tracker",(*lvcite)->GetName().c_str()) == 0) tracker.push_back(*lvcite);
        if (strcmp("BEAM",(*lvcite)->GetName().substr(0,4).c_str()) == 0) tracker.push_back(*lvcite);
      }
      if (savePDandCinCalo || killInCalo || killInCaloEfH ) {
        if (strcmp("CALO",(*lvcite)->GetName().c_str()) == 0) calo.push_back(*lvcite);
        if (strcmp("VCAL",(*lvcite)->GetName().c_str()) == 0) calo.push_back(*lvcite);
      }
      if (savePDandCinMuon) {
        if (strcmp("MUON",(*lvcite)->GetName().c_str()) == 0) muon.push_back(*lvcite);
      }
    }
    edm::LogInfo("SimG4CoreApplication") << "# of LV for Tracker " 
					 << tracker.size() << " for Calo "
                                         << calo.size() << " for Muon "
					 << muon.size();
    for (unsigned int i=0; i<tracker.size(); ++i)
      edm::LogInfo("SimG4CoreApplication") << "Tracker vol " << i << " name "
					   << tracker[i]->GetName();
    for (unsigned int i=0; i<calo.size(); ++i)
      edm::LogInfo("SimG4CoreApplication") << "Calorimeter vol " <<i <<" name "
					   << calo[i]->GetName();
    for (unsigned int i=0; i<muon.size(); ++i)
      edm::LogInfo("SimG4CoreApplication") << "Muon vol " << i << " name "
					   << muon[i]->GetName();
  }

  const G4RegionStore * rs = G4RegionStore::GetInstance();
  unsigned int num = maxTimeNames.size();
  if (num > 0) {
    std::vector<double> tofs;
    if (rs) {
      std::vector<G4Region*>::const_iterator rcite;
      for (rcite = rs->begin(); rcite != rs->end(); rcite++) {
	if ((nRusRoEcal < 1.0 || pRusRoEcal < 1.0) && 
	    (*rcite)->GetName() == "EcalRegion") {  regionEcal = (*rcite); }
	if ((nRusRoHcal < 1.0 || pRusRoHcal < 1.0) && 
	    (*rcite)->GetName() == "HcalRegion") {  regionHcal = (*rcite); }
	for (unsigned int i=0; i<num; i++) {
	  if ((*rcite)->GetName() == (G4String)(maxTimeNames[i])) {
	    maxTimeRegions.push_back(*rcite);
	    tofs.push_back(maxTrackTimes[i]);
	    break;
	  }
	}
	if (tofs.size() == num) break;
      }
    }
    for (unsigned int i=0; i<tofs.size(); i++) {
      maxTrackTimes[i] = tofs[i];
      G4String name = "Unknown";
      if (maxTimeRegions[i]) name = maxTimeRegions[i]->GetName();
      edm::LogInfo("SimG4CoreApplication") << name << " with pointer " 
					   << maxTimeRegions[i]<<" KE cut off "
					   << maxTrackTimes[i];
    }
  }
}

bool StackingAction::isThisVolume(const G4VTouchable* touch, 
				  std::vector<G4LogicalVolume*> & lvs) const {

  bool flag = false;
  if (lvs.size() > 0 && touch !=0) {
    int level = ((touch->GetHistoryDepth())+1);
    if (level >= 3) {
      unsigned int  ii = (unsigned int)(level - 3);
      flag    = (std::count(lvs.begin(),lvs.end(),(touch->GetVolume(ii)->GetLogicalVolume())) != 0);
    }
  }
  return flag;
}

int StackingAction::isItPrimaryDecayProductOrConversion(const G4Track * aTrack,
							const G4Track & mother) const {

  int flag = 0;
  TrackInformationExtractor extractor;
  const TrackInformation & motherInfo(extractor(mother));
  // Check whether mother is a primary
  if (motherInfo.isPrimary()) {
    if (aTrack->GetCreatorProcess()->GetProcessType() == fDecay) flag = 1;
    else if (aTrack->GetCreatorProcess()->GetProcessType() == fElectromagnetic &&
	     aTrack->GetCreatorProcess()->GetProcessSubType() == fGammaConversion) flag = 2;
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

bool StackingAction::isItLongLived(const G4Track * aTrack) const {

  bool   flag = false;
  double time = (aTrack->GetGlobalTime())/nanosecond;
  double tofM = maxTrackTime;
  if (maxTimeRegions.size() > 0) {
    G4Region* reg = aTrack->GetTouchable()->GetVolume(0)->GetLogicalVolume()->GetRegion();
    for (unsigned int i=0; i<maxTimeRegions.size(); i++) {
      if (reg == maxTimeRegions[i]) {
	tofM = maxTrackTimes[i];
	break;
      }
    }
  }
  if (time > tofM) flag = true;
  return flag;
}


