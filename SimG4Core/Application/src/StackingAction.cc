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
  killBeamPipe   = p.getParameter<bool>("KillBeamPipe");
  limitEnergyForVacuum = p.getParameter<double>("CriticalEnergyForVacuum")*MeV;
  maxTrackTime   = p.getParameter<double>("MaxTrackTime")*ns;
  maxTrackTimes  = p.getParameter<std::vector<double> >("MaxTrackTimes");
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
    killInCalo = (p.getParameter<edm::ParameterSet>("TestKillingOptions"))
      .getParameter<bool>("KillInCalo");
    killInCaloEfH = (p.getParameter<edm::ParameterSet>("TestKillingOptions"))
      .getParameter<bool>("KillInCaloEfH");
    edm::LogWarning("SimG4CoreApplication") 
      << " *** Activating special test killing options in StackingAction \n"
      << " *** Kill secondaries in Calorimetetrs volume = " << killInCalo << "\n"
      << " *** Kill electromagnetic secondaries from hadrons in Calorimeters volume= "
      << killInCaloEfH;
  }

  initPointer();
  newTA = new NewTrackAction();

  edm::LogInfo("SimG4CoreApplication") << "StackingAction initiated with"
				       << " flag for saving decay products in "
				       << " Tracker: " << savePDandCinTracker
                                       << " in Calo: " << savePDandCinCalo
                                       << " in Muon: " << savePDandCinMuon
                                       << " everywhere: " << savePDandCinAll
				       << "\n               saveFirstSecondary"
				       << ": " << saveFirstSecondary
				       << " Flag for tracking neutrino: "
				       << trackNeutrino 
				       << " Kill Delta Ray flag: "
				       << killDeltaRay
				       << " Killing Flag for hadrons/ions: "
				       << killHeavy;

  if(killHeavy) {
    edm::LogInfo("SimG4CoreApplication") << "StackingAction kill protons below " 
					 << kmaxProton/MeV <<" MeV, neutrons below "
					 << kmaxNeutron/MeV << " MeV and ions"
					 << " below " << kmaxIon/MeV << " MeV";
  }

  edm::LogInfo("SimG4CoreApplication") << "StackingAction kill tracks with "
				       << "time larger than " << maxTrackTime/ns
				       << " ns ";
  numberTimes = maxTimeRegions.size(); 
  if(0 < numberTimes) {
    for (unsigned int i=0; i<numberTimes; ++i) {
      edm::LogInfo("SimG4CoreApplication") << "StackingAction MaxTrackTime for "
					   << maxTimeNames[i] << " is " 
					   << maxTrackTimes[i] << " ns ";
      maxTrackTimes[i] *= ns;
    }
  }
  if(gRRactive) {
    edm::LogInfo("SimG4CoreApplication") 
      << "StackingAction: "
      << "Russian Roulette for gamma Elimit(MeV)= " 
      << gRusRoEnerLim/MeV << "\n"
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

  if(savePDandCinTracker) {
    edm::LogInfo("SimG4CoreApplication") << "StackingAction Tracker regions ";
    printRegions(trackerRegions); 
  }
  if(savePDandCinCalo) {
    edm::LogInfo("SimG4CoreApplication") << "StackingAction Calo regions ";
    printRegions(caloRegions); 
  }
  if(savePDandCinMuon) {
    edm::LogInfo("SimG4CoreApplication") << "StackingAction Muon regions ";
    printRegions(trackerRegions); 
  }
  if(killBeamPipe) {
    edm::LogInfo("SimG4CoreApplication") 
      << "StackingAction Dead regions - kill if E(MeV) < " 
      << limitEnergyForVacuum/MeV;
    printRegions(deadRegions); 
  }
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
    double ke  = aTrack->GetKineticEnergy();

    if (aTrack->GetTrackStatus() == fStopAndKill) { classification = fKill; }
    if (killHeavy && classification != fKill) {
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

    if (killDeltaRay && classification != fKill 
	&& aTrack->GetCreatorProcess()->GetProcessSubType() == fIonisation) {
      classification = fKill;
    }

    const G4Region* reg = aTrack->GetVolume()->GetLogicalVolume()->GetRegion();
    if (killInCalo && classification != fKill 
	&& isThisRegion(reg, caloRegions)) {
      classification = fKill; 
    }
    if (classification != fKill && ke <= limitEnergyForVacuum && killBeamPipe 
	&& isThisRegion(reg, deadRegions)) {
      classification = fKill; 
    }

    // Russian roulette && MC truth
    if(classification != fKill) {
      const G4Track * mother = CurrentG4Track::track();
      int flag = 0;
      if(savePDandCinAll) {
	flag = isItPrimaryDecayProductOrConversion(aTrack, *mother);
      } else {
	if ((savePDandCinTracker && isThisRegion(reg,trackerRegions)) ||
	    (savePDandCinCalo    && isThisRegion(reg,caloRegions)) ||
	    (savePDandCinMuon    && isThisRegion(reg,muonRegions))) {
	  flag = isItPrimaryDecayProductOrConversion(aTrack, *mother);
	}
      }
      if (saveFirstSecondary && 0 == flag) { 
	flag = isItFromPrimary(*mother, flag); 
      }

      // Russian roulette
      if(2112 == pdg || 22 == pdg || 2212 == pdg) {
	double currentWeight = aTrack->GetWeight();

	if(1.0 >= currentWeight) {
	  double prob = 1.0;
	  double elim = 0.0;

	  // neutron
	  if(nRRactive && pdg == 2112) {
	    elim = nRusRoEnerLim;
	    if(reg == regionEcal)             { prob = nRusRoEcal; }
	    else if(reg == regionHcal)        { prob = nRusRoHcal; }
	    else if(reg == regionMuonIron)    { prob = nRusRoMuonIron; }
	    else if(reg == regionPreShower)   { prob = nRusRoPreShower; }
	    else if(reg == regionCastor)      { prob = nRusRoCastor; }
	    else if(reg == regionWorld)       { prob = nRusRoWorld; }

	    // gamma
	  } else if(gRRactive && pdg == 22) {
	    elim = gRusRoEnerLim;
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
	  if ( isThisRegion(reg,caloRegions)) { 
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

void StackingAction::initPointer() 
{
  // Russian roulette
  const G4RegionStore * rs = G4RegionStore::GetInstance();
  std::vector<G4Region*>::const_iterator rcite;
  for (rcite = rs->begin(); rcite != rs->end(); rcite++) {
    const G4Region* reg = (*rcite);
    G4String rname = reg->GetName(); 
    if ((gRusRoEcal < 1.0 || nRusRoEcal < 1.0 || pRusRoEcal < 1.0) && 
	rname == "EcalRegion") {  
      regionEcal = reg; 
    }
    if ((gRusRoHcal < 1.0 || nRusRoHcal < 1.0 || pRusRoHcal < 1.0) && 
	rname == "HcalRegion") {  
      regionHcal = reg; 
    }
    if ((gRusRoMuonIron < 1.0 || nRusRoMuonIron < 1.0 || pRusRoMuonIron < 1.0) && 
	rname == "MuonIron") {  
      regionMuonIron = reg; 
    }
    if ((gRusRoPreShower < 1.0 || nRusRoPreShower < 1.0 || pRusRoPreShower < 1.0) 
	&& rname == "PreshowerRegion") {  
      regionPreShower = reg; 
    }
    if ((gRusRoCastor < 1.0 || nRusRoCastor < 1.0 || pRusRoCastor < 1.0) && 
	rname == "CastorRegion") {  
      regionCastor = reg; 
    }
    if ((nRusRoWorld < 1.0 || nRusRoWorld < 1.0 || pRusRoWorld < 1.0) && 
	rname == "DefaultRegionForTheWorld") {  
      regionWorld = reg; 
    }

    // time limits
    unsigned int num = maxTimeNames.size();
    maxTimeRegions.resize(num, 0);
    if (num > 0) {
      for (unsigned int i=0; i<num; i++) {
	if (rname == (G4String)(maxTimeNames[i])) {
	  maxTimeRegions[i] = reg;
	  break;
	}
      }
    }
    // 
    if(rname == "BeamPipeVacuum") {
      if(savePDandCinTracker) { trackerRegions.push_back(reg); }
      deadRegions.push_back(reg);
    }  
    if(savePDandCinTracker && 
       (rname == "BeamPipe" || rname == "TrackerPixelSensRegion"
	|| rname == "TrackerPixelDeadRegion" 
	|| rname == "TrackerDeadRegion" || rname == "TrackerSensRegion")) {
      trackerRegions.push_back(reg);
    }
    if(savePDandCinCalo && 
       (rname == "HcalRegion" || rname == "EcalRegion" 
	|| rname == "PreshowerSensRegion" || rname == "PreshowerRegion")) {
      caloRegions.push_back(reg);
    }
    if(savePDandCinMuon &&
       (rname == "MuonChamber" || rname == "MuonSensitive_RPC" 
	|| rname == "MuonIron" || rname == "Muon" 
	|| rname == "MuonSensitive_DT-CSC") ) {
      muonRegions.push_back(reg);
    }
    if(rname == "BeamPipeOutside" || rname == "QuadRegion") {
      deadRegions.push_back(reg);
    }
  }
}

bool StackingAction::isThisRegion(const G4Region* reg, 
				  std::vector<const G4Region*>& regions) const
{
  bool flag = false;
  unsigned int nRegions = regions.size();
  for(unsigned int i=0; i<nRegions; ++i) {
    if(reg == regions[i]) {
      flag = true;
      break;
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

int StackingAction::isItFromPrimary(const G4Track & mother, int flagIn) const 
{
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
  double tofM = maxTrackTime;
  if (numberTimes > 0) {
    G4Region* reg = 
      aTrack->GetVolume()->GetLogicalVolume()->GetRegion();
    for (unsigned int i=0; i<numberTimes; ++i) {
      if (reg == maxTimeRegions[i]) {
	tofM = maxTrackTimes[i];
	break;
      }
    }
  }
  if (aTrack->GetGlobalTime() > tofM) { flag = true; }
  return flag;
}

void StackingAction::printRegions(const std::vector<const G4Region*>& reg) const 
{
  G4ExceptionDescription ed;
   
  for (unsigned int i=0; i<reg.size(); ++i) {
    ed << "           " << i << ". " << reg[i]->GetName() << "\n";
  }
  edm::LogInfo("SimG4CoreApplication") << ed;
}
