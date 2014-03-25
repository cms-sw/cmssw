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
#include "G4SystemOfUnits.hh"

#include<algorithm>

//#define DebugLog

StackingAction::StackingAction(const edm::ParameterSet & p) 
{
  trackNeutrino  = p.getParameter<bool>("TrackNeutrino");
  killHeavy      = p.getParameter<bool>("KillHeavy");
  kmaxIon        = p.getParameter<double>("IonThreshold")*MeV;
  kmaxProton     = p.getParameter<double>("ProtonThreshold")*MeV;
  kmaxNeutron    = p.getParameter<double>("NeutronThreshold")*MeV;
  killDeltaRay   = p.getParameter<bool>("KillDeltaRay");
  maxTrackTime   = p.getParameter<double>("MaxTrackTime")*ns;
  maxTrackTimes  = p.getParameter<std::vector<double> >("MaxTrackTimes");
  for(unsigned int i=0; i<maxTrackTimes.size(); ++i) { maxTrackTimes[i] *= ns; }
  maxTimeNames   = p.getParameter<std::vector<std::string> >("MaxTimeNames");
  savePDandCinAll = 
    p.getUntrackedParameter<bool>("SaveAllPrimaryDecayProductsAndConversions",
				  true);
  savePDandCinTracker = 
    p.getUntrackedParameter<bool>("SavePrimaryDecayProductsAndConversionsInTracker",
				  false);
  savePDandCinCalo = 
    p.getUntrackedParameter<bool>("SavePrimaryDecayProductsAndConversionsInCalo",
				  false);
  savePDandCinMuon = 
    p.getUntrackedParameter<bool>("SavePrimaryDecayProductsAndConversionsInMuon",
				  false);
  saveFirstSecondary = 
    p.getUntrackedParameter<bool>("SaveFirstLevelSecondary",false);
  killInCalo = false;
  killInCaloEfH = false;

  // Russian Roulette
  regionEcal = 0;
  regionHcal = 0;
  regionMuonIron = 0;
  regionPreShower= 0;
  regionCastor = 0;
  regionWorld = 0;

  gRusRoEnerLim = p.getParameter<double>("RusRoGammaEnergyLimit")*MeV;
  nRusRoEnerLim = p.getParameter<double>("RusRoNeutronEnergyLimit")*MeV;
  pRusRoEnerLim = p.getParameter<double>("RusRoProtonEnergyLimit")*MeV;

  gRusRoEcal = p.getParameter<double>("RusRoEcalGamma");
  gRusRoHcal = p.getParameter<double>("RusRoHcalGamma");
  gRusRoMuonIron = p.getParameter<double>("RusRoMuonIronGamma");
  gRusRoPreShower = p.getParameter<double>("RusRoPreShowerGamma");
  gRusRoCastor = p.getParameter<double>("RusRoCastorGamma");
  gRusRoWorld = p.getParameter<double>("RusRoWorldGamma");

  nRusRoEcal = p.getParameter<double>("RusRoEcalNeutron");
  nRusRoHcal = p.getParameter<double>("RusRoHcalNeutron");
  nRusRoMuonIron = p.getParameter<double>("RusRoMuonIronNeutron");
  nRusRoPreShower = p.getParameter<double>("RusRoPreShowerNeutron");
  nRusRoCastor = p.getParameter<double>("RusRoCastorNeutron");
  nRusRoWorld = p.getParameter<double>("RusRoWorldNeutron");

  pRusRoEcal = p.getParameter<double>("RusRoEcalProton");
  pRusRoHcal = p.getParameter<double>("RusRoHcalProton");
  pRusRoMuonIron = p.getParameter<double>("RusRoMuonIronProton");
  pRusRoPreShower = p.getParameter<double>("RusRoPreShowerProton");
  pRusRoCastor = p.getParameter<double>("RusRoCastorProton");
  pRusRoWorld = p.getParameter<double>("RusRoWorldProton");

  gRRactive = false;
  nRRactive = false;
  pRRactive = false;
  if(gRusRoEnerLim > 0.0 && 
     (gRusRoEcal < 1.0 || gRusRoHcal < 1.0 || 
      gRusRoMuonIron < 1.0 || gRusRoPreShower < 1.0 || gRusRoCastor < 1.0 ||
      gRusRoWorld < 1.0)) { gRRactive = true; }
  if(nRusRoEnerLim > 0.0 && 
     (nRusRoEcal < 1.0 || nRusRoHcal < 1.0 || 
      nRusRoMuonIron < 1.0 || nRusRoPreShower < 1.0 || nRusRoCastor < 1.0 ||
      nRusRoWorld < 1.0)) { nRRactive = true; }
  if(pRusRoEnerLim > 0.0 && 
     (pRusRoEcal < 1.0 || pRusRoHcal < 1.0 || 
      pRusRoMuonIron < 1.0 || pRusRoPreShower < 1.0 || pRusRoCastor < 1.0 ||
      pRusRoWorld < 1.0)) { pRRactive = true; }

  if ( p.exists("TestKillingOptions") ) {
    killInCalo = 
      (p.getParameter<edm::ParameterSet>("TestKillingOptions")).getParameter<bool>("KillInCalo");
    killInCaloEfH = 
      (p.getParameter<edm::ParameterSet>("TestKillingOptions")).getParameter<bool>("KillInCaloEfH");
    edm::LogWarning("SimG4CoreApplication") 
      << " *** Activating special test killing options in StackingAction \n"
      << " *** Kill secondaries in Calorimetetrs volume = " << killInCalo << "\n"
      << " *** Kill electromagnetic secondaries from hadrons in Calorimeters volume = " 
      << killInCaloEfH;
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
  if(gRRactive) {
    edm::LogInfo("SimG4CoreApplication") 
      << "StackingAction: "
      << "Russian Roulette for gamma Elimit(MeV)= " 
      << nRusRoEnerLim/MeV << "\n"
      << "                 ECAL Prob= " << gRusRoEcal << "\n"
      << "                 HCAL Prob= " << gRusRoHcal << "\n"
      << "             MuonIron Prob= " << gRusRoMuonIron << "\n"
      << "            PreShower Prob= " << gRusRoPreShower << "\n"
      << "               CASTOR Prob= " << gRusRoCastor << "\n"
      << "                World Prob= " << gRusRoWorld;
  }
  if(nRRactive) {
    edm::LogInfo("SimG4CoreApplication") 
      << "StackingAction: "
      << "Russian Roulette for neutron Elimit(MeV)= " 
      << nRusRoEnerLim/MeV << "\n"
      << "                 ECAL Prob= " << nRusRoEcal << "\n"
      << "                 HCAL Prob= " << nRusRoHcal << "\n"
      << "             MuonIron Prob= " << nRusRoMuonIron << "\n"
      << "            PreShower Prob= " << nRusRoPreShower << "\n"
      << "               CASTOR Prob= " << nRusRoCastor << "\n"
      << "                World Prob= " << nRusRoWorld;
  }
  if(pRRactive) {
    edm::LogInfo("SimG4CoreApplication") 
      << "StackingAction: "
      << "Russian Roulette for proton Elimit(MeV)= " 
      << pRusRoEnerLim/MeV << "\n"
      << "                 ECAL Prob= " << pRusRoEcal << "\n"
      << "                 HCAL Prob= " << pRusRoHcal << "\n"
      << "             MuonIron Prob= " << pRusRoMuonIron << "\n"
      << "            PreShower Prob= " << pRusRoPreShower << "\n"
      << "               CASTOR Prob= " << pRusRoCastor << "\n"
      << "                World Prob= " << pRusRoWorld;
  }
  initPointer();
  newTA = new NewTrackAction();
}

StackingAction::~StackingAction() 
{
  delete newTA;
}

G4ClassificationOfNewTrack StackingAction::ClassifyNewTrack(const G4Track * aTrack) 
{
  // G4 interface part
  G4ClassificationOfNewTrack classification = fUrgent;
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
    newTA->primary(aTrack);
    /*
    if (!trackNeutrino) {
      int pdg = std::abs(aTrack->GetDefinition()->GetPDGEncoding());
      if (pdg == 12 || pdg == 14 || pdg == 16 || pdg == 18) {
	classification = fKill;
      }
    }
    */
  } else if (aTrack->GetTouchable() == 0) {
    edm::LogError("SimG4CoreApplication")
      << "StackingAction: no touchable for track " << aTrack->GetTrackID()
      << " from " << aTrack->GetParentID()
      << " with PDG code " << aTrack->GetDefinition()->GetParticleName();
    classification = fKill;
  } else {
    int pdg = aTrack->GetDefinition()->GetPDGEncoding();

    if (aTrack->GetTrackStatus() == fStopAndKill) { classification = fKill; }
    if (killHeavy && classification != fKill) {
      double ke  = aTrack->GetKineticEnergy()/MeV;
      if (((pdg/1000000000 == 1) && (((pdg/10000)%100) > 0) && 
	   (((pdg/10)%100) > 0) && (ke<kmaxIon)) || 
	  ((pdg == 2212) && (ke < kmaxProton)) ||
	  ((pdg == 2112) && (ke < kmaxNeutron))) { classification = fKill; }
    }
    if (!trackNeutrino  && classification != fKill) {
      if (pdg == 12 || pdg == 14 || pdg == 16 || pdg == 18) 
	classification = fKill;
    }
    if (classification != fKill && isItLongLived(aTrack)) 
      { classification = fKill; }
    if (killDeltaRay && classification != fKill) {
      if (aTrack->GetCreatorProcess()->GetProcessType() == fElectromagnetic &&
	  aTrack->GetCreatorProcess()->GetProcessSubType() == fIonisation)
	classification = fKill;
    }
    if (killInCalo && classification != fKill) {
      if ( isThisVolume(aTrack->GetTouchable(),calo)) { 
        classification = fKill; 
      }
    }
    // Russian roulette && MC truth
    if(classification != fKill) {
      const G4Track * mother = CurrentG4Track::track();
      int flag = 0;
      if(savePDandCinAll) {
	flag = isItPrimaryDecayProductOrConversion(aTrack, *mother);
      } else {
	if ((savePDandCinTracker && isThisVolume(aTrack->GetTouchable(),tracker)) ||
	    (savePDandCinCalo    && isThisVolume(aTrack->GetTouchable(),calo)) ||
	    (savePDandCinMuon    && isThisVolume(aTrack->GetTouchable(),muon))) {
	  flag = isItPrimaryDecayProductOrConversion(aTrack, *mother);
	}
      }
      if (saveFirstSecondary) { flag = isItFromPrimary(*mother, flag); }

      // Russian roulette
      if(2112 == pdg || 22 == pdg || 2212 == pdg) {
	double currentWeight = aTrack->GetWeight();

	if(1.0 >= currentWeight) {
	  double prob = 1.0;
	  double elim = 0.0;

	  // neutron
	  if(nRRactive && pdg == 2112) {
	    elim = nRusRoEnerLim;
	    G4Region* reg = aTrack->GetVolume()->GetLogicalVolume()->GetRegion();
	    if(reg == regionEcal)             { prob = nRusRoEcal; }
	    else if(reg == regionHcal)        { prob = nRusRoHcal; }
	    else if(reg == regionMuonIron)    { prob = nRusRoMuonIron; }
	    else if(reg == regionPreShower)   { prob = nRusRoPreShower; }
	    else if(reg == regionCastor)      { prob = nRusRoCastor; }
	    else if(reg == regionWorld)       { prob = nRusRoWorld; }

	    // gamma
	  } else if(gRRactive && pdg == 22) {
	    elim = gRusRoEnerLim;

	    G4Region* reg = aTrack->GetVolume()->GetLogicalVolume()->GetRegion();
	    if(reg == regionEcal || reg == regionPreShower) {
              if(rrApplicable(aTrack, *mother)) {
		if(reg == regionEcal)         { prob = gRusRoEcal; }
                else                          { prob = gRusRoPreShower; }
	      }
	    } else {
	      if(reg == regionHcal)           { prob = gRusRoHcal; }
	      else if(reg == regionMuonIron)  { prob = gRusRoMuonIron; }
	      else if(reg == regionCastor)    { prob = gRusRoCastor; }
	      else if(reg == regionWorld)     { prob = gRusRoWorld; }
	    }

	    // proton
	  } else if(pRRactive && pdg == 2212) {
	    elim = pRusRoEnerLim;
	    G4Region* reg = aTrack->GetVolume()->GetLogicalVolume()->GetRegion();
	    if(reg == regionEcal)             { prob = pRusRoEcal; }
	    else if(reg == regionHcal)        { prob = pRusRoHcal; }
	    else if(reg == regionMuonIron)    { prob = pRusRoMuonIron; }
	    else if(reg == regionPreShower)   { prob = pRusRoPreShower; }
	    else if(reg == regionCastor)      { prob = pRusRoCastor; }
	    else if(reg == regionWorld)       { prob = pRusRoWorld; }
	  }
	  if(prob < 1.0 && aTrack->GetKineticEnergy() < elim) {
	    if(G4UniformRand() < prob) {
	      const_cast<G4Track*>(aTrack)->SetWeight(currentWeight/prob);
	    } else {
	      classification = fKill;
	    }
	  }
	}
      }
	
      if(classification != fKill && killInCaloEfH) {
	int pdgMother = mother->GetDefinition()->GetPDGEncoding();
	if ( (pdg == 22 || std::abs(pdg) == 11) && 
	     (std::abs(pdgMother) < 11 || std::abs(pdgMother) > 17) && 
	     pdgMother != 22  ) {
	  if ( isThisVolume(aTrack->GetTouchable(),calo)) { 
	    classification = fKill; 
	  }
	}
      }
      
      if (classification != fKill) {
	newTA->secondary(aTrack, *mother, flag);
      }
    }
  /*
  double wt2 = aTrack->GetWeight();
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
  }
#ifdef DebugLog
  LogDebug("SimG4CoreApplication") << "StackingAction:Classify Track "
				   << aTrack->GetTrackID() << " Parent " 
				   << aTrack->GetParentID() << " Type "
				   << aTrack->GetDefinition()->GetParticleName() 
				   << " K.E. " << aTrack->GetKineticEnergy()/MeV
				   << " MeV from process/subprocess " 
				   << aTrack->GetCreatorProcess()->GetProcessType() 
				   << "|"
				   <<aTrack->GetCreatorProcess()->GetProcessSubType()
				   << " as " << classification << " Flag " << flag;
#endif
  
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
        if (strcmp("Tracker",(*lvcite)->GetName().c_str()) == 0) {
	  tracker.push_back(*lvcite);
	}
        if (strcmp("BEAM",(*lvcite)->GetName().substr(0,4).c_str()) == 0) {
	  tracker.push_back(*lvcite);
	}
      }
      if (savePDandCinCalo || killInCalo || killInCaloEfH ) {
        if (strcmp("CALO",(*lvcite)->GetName().c_str()) == 0) {
	  calo.push_back(*lvcite);
	}
        if (strcmp("VCAL",(*lvcite)->GetName().c_str()) == 0) {
	  calo.push_back(*lvcite);
	}
      }
      if (savePDandCinMuon) {
        if (strcmp("MUON",(*lvcite)->GetName().c_str()) == 0) {
	  muon.push_back(*lvcite);
	}
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

  std::vector<double> tofs;

  // Russian roulette
  const G4RegionStore * rs = G4RegionStore::GetInstance();
  std::vector<G4Region*>::const_iterator rcite;
  for (rcite = rs->begin(); rcite != rs->end(); rcite++) {
    if ((gRusRoEcal < 1.0 || nRusRoEcal < 1.0 || pRusRoEcal < 1.0) && 
	(*rcite)->GetName() == "EcalRegion") {  
      regionEcal = (*rcite); 
    }
    if ((gRusRoHcal < 1.0 || nRusRoHcal < 1.0 || pRusRoHcal < 1.0) && 
	(*rcite)->GetName() == "HcalRegion") {  
      regionHcal = (*rcite); 
    }
    if ((gRusRoMuonIron < 1.0 || nRusRoMuonIron < 1.0 || pRusRoMuonIron < 1.0) && 
	(*rcite)->GetName() == "MuonIron") {  
      regionMuonIron = (*rcite); 
    }
    if ((gRusRoPreShower < 1.0 || nRusRoPreShower < 1.0 || pRusRoPreShower < 1.0) 
	&& (*rcite)->GetName() == "PreshowerRegion") {  
      regionPreShower = (*rcite); 
    }
    if ((gRusRoCastor < 1.0 || nRusRoCastor < 1.0 || pRusRoCastor < 1.0) && 
	(*rcite)->GetName() == "CastorRegion") {  
      regionCastor = (*rcite); 
    }
    if ((nRusRoWorld < 1.0 || nRusRoWorld < 1.0 || pRusRoWorld < 1.0) && 
	(*rcite)->GetName() == "DefaultRegionForTheWorld") {  
      regionWorld = (*rcite); 
    }

    // time limits
    unsigned int num = maxTimeNames.size();
    if (num > 0) {
      for (unsigned int i=0; i<num; i++) {
	if ((*rcite)->GetName() == (G4String)(maxTimeNames[i])) {
	  maxTimeRegions.push_back(*rcite);
	  tofs.push_back(maxTrackTimes[i]);
	  break;
	}
      }
    }
  }
  if(0 < tofs.size()) {
    for (unsigned int i=0; i<tofs.size(); i++) {
      maxTrackTimes[i] = tofs[i];
      G4String name = "Unknown";
      if (maxTimeRegions[i]) { name = maxTimeRegions[i]->GetName(); }
      edm::LogInfo("SimG4CoreApplication") << name << " with pointer " 
					   << maxTimeRegions[i]<<" KE cut off "
					   << maxTrackTimes[i];
    }
  }
}

bool StackingAction::isThisVolume(const G4VTouchable* touch, 
				  std::vector<G4LogicalVolume*> & lvs) const 
{
  bool flag = false;
  if (lvs.size() > 0 && touch !=0) {
    int level = ((touch->GetHistoryDepth())+1);
    if (level >= 3) {
      unsigned int  ii = (unsigned int)(level - 3);
      flag = (std::count(lvs.begin(),lvs.end(),
			 (touch->GetVolume(ii)->GetLogicalVolume())) != 0);
    }
  }
  return flag;
}

int StackingAction::isItPrimaryDecayProductOrConversion(const G4Track * aTrack,
							const G4Track & mother) const
{
  int flag = 0;
  TrackInformationExtractor extractor;
  const TrackInformation & motherInfo(extractor(mother));
  // Check whether mother is a primary
  if (motherInfo.isPrimary()) {
    if (aTrack->GetCreatorProcess()->GetProcessType() == fDecay) {
      flag = 1;
    } else if (aTrack->GetCreatorProcess()->GetProcessSubType()==fGammaConversion) {
      flag = 2;
      //    } else {
      // flag = 3;
    } 
  }   
  return flag;
}

bool StackingAction::rrApplicable(const G4Track * aTrack,
				  const G4Track & mother) const
{
  bool flag = true;
  TrackInformationExtractor extractor;
  const TrackInformation & motherInfo(extractor(mother));
  // Check whether mother is a primary
  //if (motherInfo.isPrimary()) {
  // }   
  int genID = motherInfo.genParticlePID();
  if(22 == genID || 11 == genID || -11 == genID) { flag = false; }
    
  /*
  //check if the primary was g, e+, e-
  int genID = motherInfo.genParticlePID();
  double genp = motherInfo.genParticleP();
  std::cout << "Track# " << aTrack->GetTrackID() << "  " 
	    << aTrack->GetDefinition()->GetParticleName()  
	    << "  E(MeV)= " << aTrack->GetKineticEnergy()/MeV 
	    << " mother: " << mother.GetTrackID()
	    << "  " << mother.GetDefinition()->GetParticleName()
	    << " E(GeV)= " <<  mother.GetKineticEnergy()/GeV
	    << " flag: " << flag << " genID= " << genID 
	    << " p(GeV)= " << genp/GeV << std::endl; 
    */
  return flag;
}

int StackingAction::isItFromPrimary(const G4Track & mother, int flagIn) const {

  int flag = flagIn;
  if (flag != 1) {
    TrackInformationExtractor extractor;
    const TrackInformation & motherInfo(extractor(mother));
    if (motherInfo.isPrimary()) { flag = 3; }
  }
  return flag;
}

bool StackingAction::isItLongLived(const G4Track * aTrack) const 
{
  bool   flag = false;
  double time = (aTrack->GetGlobalTime())/nanosecond;
  double tofM = maxTrackTime;
  if (maxTimeRegions.size() > 0) {
    G4Region* reg = 
      aTrack->GetTouchable()->GetVolume(0)->GetLogicalVolume()->GetRegion();
    for (unsigned int i=0; i<maxTimeRegions.size(); i++) {
      if (reg == maxTimeRegions[i]) {
	tofM = maxTrackTimes[i];
	break;
      }
    }
  }
  if (time > tofM) { flag = true; }
  return flag;
}


